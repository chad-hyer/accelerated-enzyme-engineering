from hpo_runner_hourglass import DynamicMetaLearner as HourglassDynamicMetaLearner
from hpo_runner_old import DynamicMetaLearner as ExpansionDynamicMetaLearner
from RUB_reverse_meta_tiling import run_ridge

import os
import argparse
import itertools
import pandas as pd
import numpy as np
import re

import torch
from meta_tensor_builder import build_inference_tensor, AA_ORDER, AA_TO_IDX

def recommend_worst_n(model_path, evc_path, prediction_path, global_start, global_end, config, DynamicMetaLearner, n=50):
    """
    Predicts and returns the 'Bottom N' mutations (Highest Median Error).
    Excludes known training data and invalid mutations.
    """
    # 1. Load Model
    model = DynamicMetaLearner(**config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 2. Build Input Tensor
    tensor_input = build_inference_tensor(
        evc_path, 
        prediction_path, 
        global_start=global_start, 
        global_end=global_end
    )
    
    # 3. Create 'Allow List' and WT Lookup
    pred_df = pd.read_csv(prediction_path)
    length = global_end - global_start + 1
    valid_candidate_mask = np.zeros((length, 20), dtype=bool)
    wt_map = {} 
    
    def parse_mut_local(s):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(s))
        if m: return m.groups()
        return None, None, None

    for _, row in pred_df.iterrows():
        val = row['AminoAcid']
        wt, pos_str, mut = parse_mut_local(val)
        if pos_str is not None:
            wt_map[int(pos_str)] = wt
        
        # STRICT FILTER: Only 'test' rows are candidates
        if str(row['Class']).strip().lower() != 'test':
            continue

        if pos_str is not None:
            pos = int(pos_str)
            if global_start <= pos <= global_end:
                pos_idx = pos - global_start
                if mut in AA_TO_IDX:
                    aa_idx = AA_TO_IDX[mut]
                    valid_candidate_mask[pos_idx, aa_idx] = True

    # 4. Predict Error Map
    with torch.no_grad():
        pred_error_map = model(tensor_input) 
        
    score_grid = pred_error_map.squeeze().numpy()
    
    # 5. Apply Masks (Set Invalid/Known to Infinity)
    score_grid[~valid_candidate_mask] = np.inf
    known_mask = tensor_input[0, 3, :, :].numpy()
    score_grid[known_mask == 1] = np.inf 
    
    # --- CHANGED LOGIC FOR BOTTOM N ---
    
    # 6. Flatten and Filter
    flat_scores = score_grid.flatten()
    
    # Get indices of all mutations that are NOT Infinity (Valid & Unknown)
    valid_indices = np.where(flat_scores != np.inf)[0]
    
    if len(valid_indices) == 0:
        print("No valid candidates found to rank.")
        return pd.DataFrame()
    
    # 7. Sort Descending (Highest Error First)
    # We sort the valid_indices based on their scores
    sorted_indices_desc = valid_indices[np.argsort(-flat_scores[valid_indices])]
    
    recommendations = []
    
    #print(f"--- Bottom {n} Mutations (Highest Predicted Error) ---")
    #print(f"{'Rank':<5} {'Mutation':<10} {'Pred. Error':<12}")
    
    count = 0
    for idx in sorted_indices_desc:
        if count >= n: break
            
        pos_idx, aa_idx = np.unravel_index(idx, score_grid.shape)
        
        real_pos = pos_idx + global_start
        mut_aa = AA_ORDER[aa_idx]
        wt_aa = wt_map.get(real_pos, "?")
        
        mutation_str = f"{wt_aa}{real_pos}{mut_aa}"
        score = flat_scores[idx]
        
        recommendations.append({
            'Rank': count + 1,
            'Mutation': mutation_str,
            'Predicted_Error': score
        })
        
        #print(f"{count+1:<5} {mutation_str:<10} {score:.5f}")
        count += 1
        
    return pd.DataFrame(recommendations)

def expansion_tile(outpath, WT_SEQUENCE, TRAIN_FILE_PATH, TEST_FILE_PATH, EC_FILE_PATH, model_path, global_start, global_end, config, total=1000, n=50):
    os.makedirs(outpath, exist_ok=True)
    prediction_path = f'{outpath}/predictions.csv'
    meta_tiling_path = f'{outpath}/tiling_path.tsv'
    real_tiling_path = f'{outpath}/meta_metrics.tsv'

    if not os.path.isfile(real_tiling_path):
        with open(real_tiling_path, 'w') as f:
                f.write("Iteration\tMedian_Error\n")

    try:
        iterations = int(total / n)
    except:
        iterations = (total + n) // n

    for i in range(iterations):
        try:
            metaDF = pd.read_csv(meta_tiling_path, delimiter='\t')
            last_it = metaDF['Iteration'].max()
        except FileNotFoundError:
            last_it = 0
            with open(meta_tiling_path, 'w') as f:
                f.write("Iteration\tBest_Mutation\tMedian_Error\n")
        
        predictions, median_error = run_ridge(
            WT_SEQUENCE = WT_SEQUENCE, 
            TRAIN_FILE_PATH = TRAIN_FILE_PATH,
            TEST_FILE_PATH = TEST_FILE_PATH, 
            EC_FILE_PATH = EC_FILE_PATH, 
            tiling_path_filename = meta_tiling_path)
        predictions.to_csv(prediction_path, index=False)
        recommendations = recommend_worst_n(
            model_path = model_path,
            evc_path = EC_FILE_PATH, 
            prediction_path = prediction_path,
            global_start = global_start, 
            global_end = global_end, 
            config = config,
            DynamicMetaLearner = ExpansionDynamicMetaLearner,
            n=n)
        
        recommendations.rename(columns={'Rank':'Iteration','Mutation':'Best_Mutation','Predicted_Error':'Median_Error'},inplace=True)
        recommendations['Iteration'] = recommendations['Iteration'] + last_it
        
        try:
            with open(meta_tiling_path, 'a') as f: # 'a' = append mode
                for index, row in recommendations.iterrows():
                    iteration_count = row['Iteration']
                    best_mutation_name = row['Best_Mutation']
                    best_median_error = row['Median_Error']
                    f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")
        
        try:
            with open(real_tiling_path, 'a') as f: # 'a' = append mode
                f.write(f"{i * n}\t{median_error}\n")
                print(f'Iteration {i*n}: {median_error}')
        except Exception as e:
            print(f"Warning: Could not write to log file {real_tiling_path}. {e}")

def hourglass_tile(outpath, WT_SEQUENCE, TRAIN_FILE_PATH, TEST_FILE_PATH, EC_FILE_PATH, model_path, global_start, global_end, config, total=1000, n=50):
    os.makedirs(outpath, exist_ok=True)
    prediction_path = f'{outpath}/predictions.csv'
    meta_tiling_path = f'{outpath}/tiling_path.tsv'
    real_tiling_path = f'{outpath}/meta_metrics.tsv'

    if not os.path.isfile(real_tiling_path):
        with open(real_tiling_path, 'w') as f:
                f.write("Iteration\tMedian_Error\n")

    try:
        iterations = int(total / n)
    except:
        iterations = (total + n) // n

    for i in range(iterations):
        try:
            metaDF = pd.read_csv(meta_tiling_path, delimiter='\t')
            last_it = metaDF['Iteration'].max()
        except FileNotFoundError:
            last_it = 0
            with open(meta_tiling_path, 'w') as f:
                f.write("Iteration\tBest_Mutation\tMedian_Error\n")
        
        predictions, median_error = run_ridge(
            WT_SEQUENCE = WT_SEQUENCE, 
            TRAIN_FILE_PATH = TRAIN_FILE_PATH,
            TEST_FILE_PATH = TEST_FILE_PATH, 
            EC_FILE_PATH = EC_FILE_PATH, 
            tiling_path_filename = meta_tiling_path)
        predictions.to_csv(prediction_path, index=False)
        recommendations = recommend_worst_n(
            model_path = model_path,
            evc_path = EC_FILE_PATH, 
            prediction_path = prediction_path,
            global_start = global_start, 
            global_end = global_end, 
            config = config,
            DynamicMetaLearner = HourglassDynamicMetaLearner,
            n=n)
        
        recommendations.rename(columns={'Rank':'Iteration','Mutation':'Best_Mutation','Predicted_Error':'Median_Error'},inplace=True)
        recommendations['Iteration'] = recommendations['Iteration'] + last_it
        
        try:
            with open(meta_tiling_path, 'a') as f: # 'a' = append mode
                for index, row in recommendations.iterrows():
                    iteration_count = row['Iteration']
                    best_mutation_name = row['Best_Mutation']
                    best_median_error = row['Median_Error']
                    f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")
        
        try:
            with open(real_tiling_path, 'a') as f: # 'a' = append mode
                f.write(f"{i * n}\t{median_error}\n")
                print(f'Iteration {i*n}: {median_error}')
        except Exception as e:
            print(f"Warning: Could not write to log file {real_tiling_path}. {e}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_idx', type=int, default=0, help='Index of config to run')
    parser.add_argument('--out_path', type=str, default='', help='Output directory for results')
    parser.add_argument('--ca_path', type=str, default='', help='Path for directory of CA files')
    parser.add_argument('--rub_path', type=str, default='', help='Path for directory of RUB files')
    parser.add_argument('--name', type=str, default='production', help='Name of output files')
    parser.add_argument('--model_type', type=str, default='expansion', help='Model architecture type ("expansion" or "hourglass")')
    parser.add_argument('--model_path', type=str, default='', help='.pth file for the model')
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
    model_save_path = args.model_path

    CA_DIR = args.ca_path#"/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files" # Directory with CA prediction_it_X.csv
    RUB_DIR = args.rub_path #"/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/RUB_files"        # Should contain ground truth xlsx
    output_base_dir = args.out_path #"/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/temp_simulations_hourglass"
    name = args.name

    if os.path.exists(model_save_path):
        print(f"Found existing model at {model_save_path}.")

    ## Tiling
    if args.model_type == 'expansion':
        print('Tiling for CA')
        expansion_tile(
            outpath = os.path.join(output_base_dir, name, 'CA'), 
            WT_SEQUENCE = 'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK', 
            TRAIN_FILE_PATH = os.path.join(CA_DIR, "CA_train.xlsx"), 
            TEST_FILE_PATH = os.path.join(CA_DIR, "CA_test.xlsx"), 
            EC_FILE_PATH = os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'), 
            model_path = model_save_path, 
            global_start = 1, 
            global_end = 243, 
            config = config, 
            total=1000, 
            n=50)
        
        print('Tiling for RUB')
        expansion_tile(
            outpath = os.path.join(output_base_dir, name, 'RUB'), 
            WT_SEQUENCE = 'MDQSSRYVNLALKEEDLIAGGEHVLCAYIMKPKAGYGYVATAAHFAAESSTGTNVEVCTTDDFTRGVDALVYEVDEARELTKIAYPVALFDRNITDGKAMIASFLTLTMGNNQGMGDVEYAKMHDFYVPEAYRALFDGPSVNISALWKVLGRPEVDGGLVVGTIIKPKLGLRPKPFAEACHAFWLGGDFIKNDEPQGNQPFAPLRDTIALVADAMRRAQDETGEAKLFSANITADDPFEIIARGEYVLETFGENASHVALLVDGYVAGAAAITTARRRFPDNFLHYHRAGHGAVTSPQSKRGYTAFVHCKMARLQGASGIHTGTMGFGKMEGESSDRAIAYMLTQDEAQGPFYRQSWGGMKACTPIISGGMNALRMPGFFENLGNANVILTAGGGAFGHIDGPVAGARSLRQAWQAWRDGVPVLDYAREHKELARAFESFPGDADQIYPGWRKALGVEDTRSALPA', 
            TRAIN_FILE_PATH = os.path.join(RUB_DIR, "RUB_train.xlsx"), 
            TEST_FILE_PATH = os.path.join(RUB_DIR, "RUB_test.xlsx"), 
            EC_FILE_PATH = os.path.join(RUB_DIR, 'RUB_single_mutant_matrix.csv'), 
            model_path = model_save_path, 
            global_start = 3, 
            global_end = 464, 
            config = config, 
            total=1000, 
            n=50)
    elif args.model_type == 'hourglass':
        print('Tiling for CA')
        hourglass_tile(
            outpath = os.path.join(output_base_dir, name, 'CA'), 
            WT_SEQUENCE = 'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK', 
            TRAIN_FILE_PATH = os.path.join(CA_DIR, "CA_train.xlsx"), 
            TEST_FILE_PATH = os.path.join(CA_DIR, "CA_test.xlsx"), 
            EC_FILE_PATH = os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'), 
            model_path = model_save_path, 
            global_start = 1, 
            global_end = 243, 
            config = config, 
            total=1000, 
            n=50)
        
        print('Tiling for RUB')
        hourglass_tile(
            outpath = os.path.join(output_base_dir, name, 'RUB'), 
            WT_SEQUENCE = 'MDQSSRYVNLALKEEDLIAGGEHVLCAYIMKPKAGYGYVATAAHFAAESSTGTNVEVCTTDDFTRGVDALVYEVDEARELTKIAYPVALFDRNITDGKAMIASFLTLTMGNNQGMGDVEYAKMHDFYVPEAYRALFDGPSVNISALWKVLGRPEVDGGLVVGTIIKPKLGLRPKPFAEACHAFWLGGDFIKNDEPQGNQPFAPLRDTIALVADAMRRAQDETGEAKLFSANITADDPFEIIARGEYVLETFGENASHVALLVDGYVAGAAAITTARRRFPDNFLHYHRAGHGAVTSPQSKRGYTAFVHCKMARLQGASGIHTGTMGFGKMEGESSDRAIAYMLTQDEAQGPFYRQSWGGMKACTPIISGGMNALRMPGFFENLGNANVILTAGGGAFGHIDGPVAGARSLRQAWQAWRDGVPVLDYAREHKELARAFESFPGDADQIYPGWRKALGVEDTRSALPA', 
            TRAIN_FILE_PATH = os.path.join(RUB_DIR, "RUB_train.xlsx"), 
            TEST_FILE_PATH = os.path.join(RUB_DIR, "RUB_test.xlsx"), 
            EC_FILE_PATH = os.path.join(RUB_DIR, 'RUB_single_mutant_matrix.csv'), 
            model_path = model_save_path, 
            global_start = 3, 
            global_end = 464, 
            config = config, 
            total=1000, 
            n=50)
    else:
        print(f'Model type {args.model_type} not recognized. Please check your commands.')
    print('Simulation Complete')

if __name__ == "__main__":
    main()