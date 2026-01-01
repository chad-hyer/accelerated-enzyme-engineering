import os
import sys
import argparse
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import itertools
import shutil
from torch.utils.data import DataLoader

# --- Imports from your provided files ---
# Ensure these files are in the same folder or PYTHONPATH
from meta_tensor_builder import ProteinTilingDataset, build_inference_tensor, AA_ORDER, AA_TO_IDX
from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel

# ==========================================
# 1. Dynamic Architecture Definition
# ==========================================
class DynamicMetaLearner(nn.Module):
    def __init__(self, input_channels=4, hidden_channels=32, kernel_size=5, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        padding_h = (kernel_size - 1) // 2 
        
        # Input Layer
        layers.append(nn.Conv2d(input_channels, hidden_channels, 
                                kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
        layers.append(nn.ReLU())
        if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
        
        # Hidden Layers
        current_channels = hidden_channels
        for _ in range(num_layers - 2):
            next_channels = current_channels * 2
            layers.append(nn.Conv2d(current_channels, next_channels, 
                                    kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            current_channels = next_channels

        # Final Projection
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Tiling Simulation Engine
# ==========================================
class TilingSimulator:
    def __init__(self, protein_name, ground_truth_train_path, ground_truth_test_path, 
                 evc_path, wt_sequence, config_id, output_base_dir, global_start, global_end):
        self.protein_name = protein_name
        self.train_original = pd.read_excel(ground_truth_train_path)
        self.test_original = pd.read_excel(ground_truth_test_path)
        self.evc_path = evc_path
        self.wt_sequence = wt_sequence
        self.global_start = global_start
        self.global_end = global_end
        
        # Create a unique workspace for this simulation to avoid file collisions
        self.sim_dir = os.path.join(output_base_dir, config_id, protein_name)
        os.makedirs(self.sim_dir, exist_ok=True)
        
        self.valid_pool = set(self.test_original['AminoAcid'].unique())

        self.tiling_path_file = os.path.join(self.sim_dir, f"{protein_name}_reversed_meta_tiling_path.tsv")
        with open(self.tiling_path_file, 'w') as f:
            f.write("Iteration\tBest_Mutation\tMedian_Error\n")

        # Pre-process EVC once (simplified from your script logic)
        self.evc_data, self.model_scope = self._preprocess_evc()

    def _preprocess_evc(self):
        # Using a simplified version of your preprocess logic for speed
        anchor_data = pd.read_csv(self.evc_path)
        # Assuming format: 'mutant' or 'Mutations', 'prediction_epistatic'
        col_map = {'mutant': 'Mutations', 'prediction_epistatic': 'Predictions'}
        anchor_data.rename(columns=col_map, inplace=True)
        
        # Create scope
        all_muts = set(anchor_data['Mutations']).union(set(self.train_original['AminoAcid'])).union(set(self.test_original['AminoAcid']))
        model_scope = sorted(list(all_muts))
        
        # Create Series
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        master_series = pd.Series(index=model_scope, dtype=float)
        master_series.update(pd.Series(anchor_data['Predictions'].values, index=anchor_data['Mutations']))
        
        # Normalize
        vals = master_series.values.reshape(-1, 1)
        master_series[:] = np.nan_to_num(sc.fit_transform(vals).flatten(), nan=0.0)
        return master_series, model_scope

    def run_ridge_step(self, current_train, current_test, iteration):
        # Initialize your Optimized Ridge Model
        model = AugmentedMLDEmodel(
            training_data=current_train,
            test_data=current_test,
            wt_sequence=self.wt_sequence,
            model_scope_mutations=self.model_scope,
            ec_predictions=self.evc_data,
            Maestro_file=None, ESM_file=None
        )
        
        # Train and Predict
        # Suppress prints for HPC logs
        # sys.stdout = open(os.devnull, 'w') 
        model.train_and_predict(model='AugmentedEC', encoding='One Hot')
        # sys.stdout = sys.__stdout__
        
        all_preds = model._predictions_df.rename(columns={'Mutation':'AminoAcid'})
        
        # Calculate Error for Test Set
        test_with_preds = current_test.merge(all_preds, on='AminoAcid', how='left')
        test_with_preds['Error'] = np.abs(test_with_preds['Prediction'] - test_with_preds['Activity'])
        median_error = test_with_preds['Error'].median()
        
        # Save prediction file for the Meta-Learner to read
        pred_path = os.path.join(self.sim_dir, f'pred_it_{iteration}.csv')
        
        # Prepare CSV for Meta-Learner (Must have Class column)
        # We need to list ALL mutations, labeling knowns as 'train' and unknowns as 'test'
        output_df = all_preds.copy()
        output_df['Class'] = 'test'
        output_df.loc[output_df['AminoAcid'].isin(current_train['AminoAcid']), 'Class'] = 'train'
        output_df.loc[output_df['Class'] == 'train', 'Activity'] = \
            output_df['AminoAcid'].map(current_train.set_index('AminoAcid')['Activity'])
            
        output_df.to_csv(pred_path, index=False)
        return median_error, pred_path

    def select_worst_n(self, meta_model, pred_path, n=50):
        # 1. Build Tensor using instance variables for start/end
        tensor_input = build_inference_tensor(
            self.evc_path, pred_path, 
            global_start=self.global_start, 
            global_end=self.global_end
        )
        
        # 2. Inference
        meta_model.eval()
        with torch.no_grad():
            output = meta_model(tensor_input)
        
        scores = output.squeeze().numpy()
        
        # 3. Mask Knowns (Train Set in Prediction File)
        mask_channel = tensor_input[0, 3, :, :].numpy()
        scores[mask_channel == 1] = -np.inf 
        
        # 4. Selection Loop
        flat_indices = np.argsort(scores.flatten())[::-1] # Descending (Worst First)
        
        selected = []
        count = 0
        
        for idx in flat_indices:
            if count >= n: break
            
            pos_idx, aa_idx = np.unravel_index(idx, scores.shape)
            # Adjust position by global_start
            real_pos = pos_idx + self.global_start
            mut_str = f"{self.wt_sequence[pos_idx]}{real_pos}{AA_ORDER[aa_idx]}"
            
            # STRICT FILTER: Only pick if it exists in our current valid pool (Test Set)
            if mut_str in self.valid_pool:
                selected.append(mut_str)
                count += 1
                
        return selected

    def run_simulation(self, meta_model, steps=10, step_size=50):
        # Initialize Working Sets
        current_train = self.train_original.copy()
        current_test = self.test_original.copy()
        
        # Reset Valid Pool to strictly be the current test set
        self.valid_pool = set(current_test['AminoAcid'].unique())
        
        history = []
        total_mutations_picked = 0
        
        for i in range(steps):
            # 1. Train Ridge & Predict
            med_error, pred_path = self.run_ridge_step(current_train, current_test, i)
            history.append({'iteration': i, 'median_error': med_error, 'train_size': len(current_train)})
            print(f"[{self.protein_name}] Step {i}: Error {med_error:.4f}, Train Size {len(current_train)}")
            
            # Stop if test set is empty
            if len(current_test) == 0:
                print("Test set exhausted.")
                break
            
            # 2. Select Next Batch via Meta-Learner
            chosen_muts = self.select_worst_n(meta_model, pred_path, n=step_size)
            
            if not chosen_muts:
                print("No valid mutations found. Stopping.")
                break
            
            # 3. Log to Tiling Path File
            with open(self.tiling_path_file, 'a') as f:
                for mut in chosen_muts:
                    total_mutations_picked += 1
                    # Format: Rank <tab> Mutation <tab> Current_Ridge_Error
                    f.write(f"{total_mutations_picked}\t{mut}\t{med_error}\n")
            
            # 4. MOVE DATA: Test -> Train (Mimics Tiling_parallelized logic)
            # Find the actual rows in current_test
            rows_to_move = current_test[current_test['AminoAcid'].isin(chosen_muts)]
            current_train = pd.concat([current_train, rows_to_move], ignore_index=True)
            current_test = current_test[~current_test['AminoAcid'].isin(chosen_muts)]
            
            # Remove from Valid Pool so they can't be picked again
            self.valid_pool -= set(chosen_muts)
            
        shutil.rmtree(self.sim_dir)
        return history

# ==========================================
# 3. Main Orchestrator
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_idx', type=int, default=0, help='Index of config to run (for SLURM Array)')
    parser.add_argument('--results_file', type=str, default='hpo_results.json')
    args = parser.parse_args()

    # --- Configuration Grid ---
    param_grid = {
        'kernel_size': [3, 5, 7, 9],
        'num_layers': [2, 3, 4],
        'hidden_channels': [16, 32],
        'dropout': [0.0, 0.2]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Select Config based on SLURM Array ID
    if args.config_idx >= len(combinations):
        print(f"Config ID {args.config_idx} out of range.")
        return
        
    config = combinations[args.config_idx]
    config_id = f"config_{args.config_idx}"
    print(f"--- Running {config_id}: {config} ---")

    # Define Model Save Path
    model_save_dir = "saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{config_id}.pth")

    # Initialize Model Architecture
    model = DynamicMetaLearner(**config)

    # --- Step 1: Check for Existing Model or Train ---
    CA_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files" # Directory with CA prediction_it_X.csv
    RUB_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/RUB_files"        # Should contain ground truth xlsx
    if os.path.exists(model_save_path):
        print(f"Found existing model at {model_save_path}. Loading weights...")
        # Load weights (map_location ensures it works even if moving GPU->CPU)
        model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    
    else:
        print(f"No existing model found for {config_id}. Starting Training...")
        
        # Define paths for Training Data (CA)
        CA_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files"
        dataset = ProteinTilingDataset(
            pred_dir=os.path.join(CA_DIR, 'predictions'),
            error_dir=os.path.join(CA_DIR, 'errors'),
            evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
            global_start=1, global_end=243 
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss(reduction='none')
        
        # Train Loop
        for epoch in range(10):
            epoch_loss = 0
            for inp, tgt, mask in dataloader:
                optimizer.zero_grad()
                output = model(inp)
                loss = (criterion(output, tgt) * mask).sum() / (mask.sum() + 1e-6)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Epoch {epoch} Loss: {epoch_loss/len(dataloader)}")

        # Save the newly trained model
        torch.save(model.state_dict(), model_save_path)
        print(f"Training complete. Model saved to {model_save_path}")

    # --- Step 1: Train Meta-Learner (Validation Task) ---
    # Define paths
    """
    CA_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files" # Directory with CA prediction_it_X.csv
    RUB_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/RUB_files"        # Should contain ground truth xlsx
    
    print("Training Meta-Learner on CA Data...")
    dataset = ProteinTilingDataset(
        pred_dir=os.path.join(CA_DIR, 'predictions'),
        error_dir=os.path.join(CA_DIR, 'errors'),
        evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
        global_start=1, global_end=243 # Adjust for CA length
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = DynamicMetaLearner(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')
    
    # Train Loop (Fixed Epochs)
    for epoch in range(10):
        for inp, tgt, mask in dataloader:
            optimizer.zero_grad()
            output = model(inp)
            loss = (criterion(output, tgt) * mask).sum() / (mask.sum() + 1e-6)
            loss.backward()
            optimizer.step()

    model_save_path = os.path.join("saved_models", f"{config_id}.pth")
    os.makedirs("saved_models", exist_ok=True) # Ensure directory exists
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    """
    # --- Step 2: Evaluate on CA (Seen) ---
    print("Evaluating on CA (Seen)...")
    ca_sim = TilingSimulator(
        protein_name="CA",
        ground_truth_train_path=os.path.join(CA_DIR, "CA_train.xlsx"),
        ground_truth_test_path=os.path.join(CA_DIR, "CA_test.xlsx"),
        evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
        wt_sequence="MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK", # CA WT
        config_id=config_id,
        output_base_dir="temp_simulations",
        global_start=1,
        global_end=243
    )
    ca_metrics = ca_sim.run_simulation(model, steps=20, step_size=50)
    
    # --- Step 3: Evaluate on RUB (Unseen) ---
    print("Evaluating on RUB (Unseen)...")
    rub_sim = TilingSimulator(
        protein_name="RUB",
        ground_truth_train_path=os.path.join(RUB_DIR, "RUB_train.xlsx"),
        ground_truth_test_path=os.path.join(RUB_DIR, "RUB_test.xlsx"),
        evc_path=os.path.join(RUB_DIR, 'RUB_single_mutant_matrix.csv'),
        wt_sequence="MDQSSRYVNLALKEEDLIAGGEHVLCAYIMKPKAGYGYVATAAHFAAESSTGTNVEVCTTDDFTRGVDALVYEVDEARELTKIAYPVALFDRNITDGKAMIASFLTLTMGNNQGMGDVEYAKMHDFYVPEAYRALFDGPSVNISALWKVLGRPEVDGGLVVGTIIKPKLGLRPKPFAEACHAFWLGGDFIKNDEPQGNQPFAPLRDTIALVADAMRRAQDETGEAKLFSANITADDPFEIIARGEYVLETFGENASHVALLVDGYVAGAAAITTARRRFPDNFLHYHRAGHGAVTSPQSKRGYTAFVHCKMARLQGASGIHTGTMGFGKMEGESSDRAIAYMLTQDEAQGPFYRQSWGGMKACTPIISGGMNALRMPGFFENLGNANVILTAGGGAFGHIDGPVAGARSLRQAWQAWRDGVPVLDYAREHKELARAFESFPGDADQIYPGWRKALGVEDTRSALPA", # RUB WT
        config_id=config_id,
        output_base_dir="temp_simulations",
        global_start=3,
        global_end=464
    )
    rub_metrics = rub_sim.run_simulation(model, steps=20, step_size=50)

    # --- Step 4: Save Results ---
    final_result = {
        'config_id': args.config_idx,
        'params': config,
        'ca_metrics': ca_metrics,
        'rub_metrics': rub_metrics
    }
    
    # Append to JSON Lines file (Thread-safe-ish for appending)
    with open(args.results_file, 'a') as f:
        f.write(json.dumps(final_result) + "\n")
        
    print(f"Config {args.config_idx} Completed.")

if __name__ == "__main__":
    main()