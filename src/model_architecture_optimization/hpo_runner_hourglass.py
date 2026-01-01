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
        MAX_CHANNELS = 64
        
        # --- ENCODER (Expansion) ---
        # We track the channel sizes to ensure perfect symmetry in the decoder
        encoder_channels = [] 
        
        current = input_channels
        next_ch = hidden_channels
        
        # Build expansion layers
        for i in range(num_layers):
            # Apply Cap
            if next_ch > MAX_CHANNELS:
                next_ch = MAX_CHANNELS
                
            # Add Block: Conv -> ReLU -> Dropout
            layers.append(nn.Conv2d(current, next_ch, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            
            # Record output of this layer for the decoder to reverse
            encoder_channels.append(next_ch)
            
            current = next_ch
            next_ch = current * 2 # Prepare next target (before capping)

        # --- DECODER (Contraction) ---
        # We reverse the list of encoder outputs to get our targets.
        # Exclude the last element (current peak) because we are already there.
        # Example: if encoder made [16, 32, 64], targets become [32, 16]
        decoder_targets = encoder_channels[:-1][::-1]
        
        for target_ch in decoder_targets:
            layers.append(nn.Conv2d(current, target_ch, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            current = target_ch

        # --- FINAL PROJECTION ---
        # Collapses the last hidden channel (e.g., 16) down to 1
        layers.append(nn.Conv2d(current, 1, kernel_size=(1, 1)))
        
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

        self.tsv_path = os.path.join(self.sim_dir, f"{protein_name}_reverse_meta_tiling_path.tsv")

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

    def select_worst_n(self, meta_model, pred_path, global_start, global_end, n=50):
        """
        Uses the meta-model to pick the N mutations with HIGHEST predicted value.
        """
        # Build Tensor
        # Determine L from EVC file or Sequence
        L = len(self.wt_sequence)
        # Assume EVC covers 1 to L or similar. We use global_start=1 for simplicity
        # Adjust if your EVC file has different indexing
        tensor_input = build_inference_tensor(self.evc_path, pred_path, global_start=global_start, global_end=global_end)
        
        # Inference
        meta_model.eval()
        with torch.no_grad():
            output = meta_model(tensor_input) # (1, 1, L, 20)
        
        scores = output.squeeze().numpy() # (L, 20)
        
        # Mask Knowns (Train set)
        mask_channel = tensor_input[0, 3, :, :].numpy()
        scores[mask_channel == 1] = -np.inf # Set knowns to -inf so they aren't picked as "High"
        
        # Also Mask Invalid (Not in Prediction File)
        # (Simplified: The tensor builder fills 0, model predicts. 
        # For strict correctness, we should mask things not in 'test' set of pred_path)
        pred_df = pd.read_csv(pred_path)
        test_muts = set(pred_df[pred_df['Class'] == 'test']['AminoAcid'])
        
        # Flatten and Sort
        flat_indices = np.argsort(scores.flatten())[::-1] # Descending (Best/Worst first)
        
        selected = []
        count = 0
        for idx in flat_indices:
            if count >= n: break
            
            # Convert to String
            pos_idx, aa_idx = np.unravel_index(idx, scores.shape)
            mut_str = f"{self.wt_sequence[pos_idx]}{pos_idx+1}{AA_ORDER[aa_idx]}"
            
            # Strict Filter: Must be in the "Test" set available to pick
            if mut_str in self.valid_pool:
                selected.append(mut_str)
                count += 1
                
        return selected

    def run_simulation(self, meta_model, steps=10, step_size=50):
        current_train = self.train_original.copy()
        current_test = self.test_original.copy()
        
        history = []

        total_mutations_picked = 0
        
        for i in range(steps):
            # 1. Ridge Step
            med_error, pred_path = self.run_ridge_step(current_train, current_test, i)
            history.append({'iteration': i, 'median_error': med_error, 'train_size': len(current_train)})
            print(f"[{self.protein_name}] It {i}: Error {med_error:.4f}, Train Size {len(current_train)}")
            
            if len(current_test) < step_size:
                print("Test set exhausted.")
                break
            
            # 2. Meta Selection
            chosen_muts = self.select_worst_n(meta_model, pred_path, global_start=self.global_start, global_end=self.global_end, n=step_size)
            
            if not chosen_muts:
                print("No valid mutations found. Stopping.")
                break
            
            # --- INSERT/REPLACE LOGGING BLOCK ---
            with open(self.tsv_path, 'a') as f:
                for mut in chosen_muts:
                    total_mutations_picked += 1
                    # Log: Rank (Sequential), Mutation, Batch Error
                    f.write(f"{total_mutations_picked}\t{mut}\t{med_error}\n")
            
            # Update Valid Pool (CRITICAL: Remove what we just picked)
            self.valid_pool -= set(chosen_muts)
            
            # 3. Update Sets
            # Move chosen from Test to Train
            new_data = current_test[current_test['AminoAcid'].isin(chosen_muts)]
            current_train = pd.concat([current_train, new_data], ignore_index=True)
            current_test = current_test[~current_test['AminoAcid'].isin(chosen_muts)]
            
        # Clean up temp files
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
        'num_layers': [2, 3, 4],     # 2=Short Hourglass, 4=Deep Hourglass
        'hidden_channels': [16, 32], # Start width
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
    model_save_dir = "saved_models_hourglass"
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
    
    # --- Step 2: Evaluate on CA (Seen) ---
    print("Evaluating on CA (Seen)...")
    ca_sim = TilingSimulator(
        protein_name="CA",
        ground_truth_train_path=os.path.join(CA_DIR, "CA_train.xlsx"),
        ground_truth_test_path=os.path.join(CA_DIR, "CA_test.xlsx"),
        evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
        wt_sequence="MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK", # CA WT
        config_id=config_id,
        output_base_dir="./temp_simulations_hourglass",
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
        output_base_dir="./temp_simulations_hourglass",
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