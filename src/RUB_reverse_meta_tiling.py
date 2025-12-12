# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:29:37 2025

@author: hyerc
"""

import os
import sys
import pandas as pd
import numpy as np

from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel
from sklearn.preprocessing import StandardScaler
import re

import torch
import torch.nn as nn
from meta_tensor_builder import build_inference_tensor, AA_ORDER, AA_TO_IDX


WT_SEQUENCE = "MDQSSRYVNLALKEEDLIAGGEHVLCAYIMKPKAGYGYVATAAHFAAESSTGTNVEVCTTDDFTRGVDALVYEVDEARELTKIAYPVALFDRNITDGKAMIASFLTLTMGNNQGMGDVEYAKMHDFYVPEAYRALFDGPSVNISALWKVLGRPEVDGGLVVGTIIKPKLGLRPKPFAEACHAFWLGGDFIKNDEPQGNQPFAPLRDTIALVADAMRRAQDETGEAKLFSANITADDPFEIIARGEYVLETFGENASHVALLVDGYVAGAAAITTARRRFPDNFLHYHRAGHGAVTSPQSKRGYTAFVHCKMARLQGASGIHTGTMGFGKMEGESSDRAIAYMLTQDEAQGPFYRQSWGGMKACTPIISGGMNALRMPGFFENLGNANVILTAGGGAFGHIDGPVAGARSLRQAWQAWRDGVPVLDYAREHKELARAFESFPGDADQIYPGWRKALGVEDTRSALPA"
TRAIN_FILE_PATH = 'RUB/RUB_train.xlsx'
TEST_FILE_PATH = 'RUB/RUB_test.xlsx'
EC_FILE_PATH = 'RUB/RUB_single_mutant_matrix.csv'
tiling_path_filename = 'D:/Downloads/Files to work with/RUB/Meta/RUB_reverse_meta_tiling_path_ss1.tsv'#'RUB/RUB_reverse_meta_tiling_path.tsv'

def preprocess_evc_data(ec_file_path, train_df, test_df):
    """
    Loads and processes the EVC data ONCE.
    This is called at the start of main() before any loops.
    """
    #print("Pre-processing EVC data once...")
    # 1. Load EVC file
    anchor_data = pd.read_csv(ec_file_path)
    anchor_data.rename(columns={'mutant': 'Mutations', 'prediction_epistatic': 'Predictions'}, inplace=True)
    evc_mutations_set = set(anchor_data['Mutations'])

    # 2. Get training mutations (after filtering WT)
    def is_wt(mutation_str):
        match = re.match(r"([A-Z])(\d+)([A-Z])", str(mutation_str))
        return match and match.group(1) == match.group(3)
    train_df_filtered = train_df[~train_df['AminoAcid'].apply(is_wt)]
    training_mutations_set = set(train_df_filtered['AminoAcid'])

    # 3. Create the master list of mutations (EVC + all train/test)
    full_dataset_mutations = set(train_df['AminoAcid']).union(set(test_df['AminoAcid']))
    model_scope_set = evc_mutations_set.union(full_dataset_mutations)
    model_scope_mutations = sorted(list(model_scope_set))

    # 4. Create the pre-processed EVC feature Series
    sc = StandardScaler()
    master_series = pd.Series(index=model_scope_mutations, dtype=float)
    raw_series = pd.Series(anchor_data['Predictions'].values, index=anchor_data['Mutations'])
    master_series.update(raw_series)

    # 5. Standardize *only* the EVC data
    non_nan_mask = master_series.notna()
    if non_nan_mask.any():
        scores_to_scale = np.array(master_series[non_nan_mask]).reshape(-1, 1)
        scaled_scores = sc.fit_transform(scores_to_scale)
        master_series[non_nan_mask] = scaled_scores.flatten()

    # 6. Fill all missing (e.g., WT) with a neutral 0.0
    master_series.fillna(0.0, inplace=True)
    
    #print("EVC data pre-processing complete.")
    return master_series, model_scope_mutations


def run_ridge(WT_SEQUENCE, TRAIN_FILE_PATH, EC_FILE_PATH, tiling_path_filename):
    # Load original data first
    try:
        train_original = pd.read_excel(TRAIN_FILE_PATH)
        test_original = pd.read_excel(TEST_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Fatal Error: Could not find original input file. {e}")
        print(f"Looked for: {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        sys.exit(1)
    
    # --- MODIFICATION 6: Pre-process EVC data ONCE ---
    # This is the key optimization
    preprocessed_ec_data, model_scope = preprocess_evc_data(EC_FILE_PATH, train_original, test_original)
    
    # --- Checkpointing Logic ---
    if os.path.exists(tiling_path_filename):
        #print(f"Found existing tiling path file. Resuming from checkpoint...")
        try:
            tiling_path_df = pd.read_csv(tiling_path_filename, sep='\t')
            if tiling_path_df.empty:
                train_main = train_original.copy()
                test_main = test_original.copy()
                iteration_count = 0
            else:
                iteration_count = tiling_path_df['Iteration'].max()
                chosen_mutations = tiling_path_df['Best_Mutation'].tolist()
                rows_to_move = test_original[test_original['AminoAcid'].isin(chosen_mutations)]
                train_main = pd.concat([train_original, rows_to_move], ignore_index=True)
                test_main = test_original[~test_original['AminoAcid'].isin(chosen_mutations)].copy()
            
            #print(f"Resuming from Iteration {iteration_count + 1}")
        except Exception as e:
            print(f"Error loading {tiling_path_filename}: {e}. Starting from scratch.")
            train_main = train_original.copy()
            test_main = test_original.copy()
            iteration_count = 0
    else:
        print("No tiling path file found. Starting from scratch.")
        train_main = train_original.copy()
        test_main = test_original.copy()
        iteration_count = 0
    
    # --- SLURM-aware core count ---
    try:
        n_cores_str = os.environ.get('SLURM_CPUS_PER_TASK')
        if n_cores_str is not None:
            n_cores = int(n_cores_str)
            #print(f"Running on SLURM. Using {n_cores} allocated cores.")
        else:
            n_cores = max(1, os.cpu_count() - 1)
            #print(f"Not on SLURM. Using {n_cores} local cores.")
    except Exception:
        n_cores = max(1, os.cpu_count() - 1)
        #print(f"Error reading SLURM_CPUS_PER_TASK. Defaulting to {n_cores} local cores.")
    
    
    model = AugmentedMLDEmodel(
        training_data=train_main, 
        test_data=test_main,          
        wt_sequence=WT_SEQUENCE,
        model_scope_mutations=model_scope, # Pass pre-processed
        ec_predictions=preprocessed_ec_data,               # Pass pre-processed
        Maestro_file=None, 
        ESM_file=None      
    )
    
    model.train_and_predict(model='AugmentedEC', encoding='One Hot')
    
    all_preds_df = model._predictions_df
    input_matrix = model.input_matrix
    train = train_main.copy()
    test = test_main.copy()
    train['Class'] = 'train'
    test['Class'] = 'test'
    whole = train._append(test,ignore_index=True)
    
    all_preds_df.rename(columns={'Mutation':'AminoAcid'},inplace=True)
    predictions = all_preds_df.merge(whole, how='left', on='AminoAcid')
    predictions['Error'] = np.abs(predictions['Prediction'] - predictions['Activity'])
    median_error = predictions[predictions['Class'] == 'test']['Error'].median()
    print(f'Median Test Error: {median_error}')
    return predictions, median_error

class TilingMetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 Channels -> Output: 1 Channel (Predicted Median Error)
        # Using Padding to keep Length dimension same
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1, 1)) # Final projection
        )
        
    def forward(self, x):
        return self.net(x)

def recommend_worst_n(model_path, evc_path, prediction_path, global_start, global_end, n=10):
    """
    Predicts and returns the 'Bottom N' mutations (Highest Median Error).
    Excludes known training data and invalid mutations.
    """
    # 1. Load Model
    model = TilingMetaLearner()
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

model_path = "meta_learner.pth"
evc_path = EC_FILE_PATH
#prediction_path = 'D:/Downloads/Files to work with/meta_model/Reversed Iteration 6 Prediction.csv'
global_start = 3
global_end = 464
meta_tiling_path = tiling_path_filename

def run_reverse_meta(model_path, evc_path, prediction_path, global_start, global_end, meta_tiling_path):
    try:
        metaDF = pd.read_csv(meta_tiling_path, delimiter='\t')
        last_it = metaDF['Iteration'].max()
    except FileNotFoundError:
        last_it = 0
        with open(meta_tiling_path, 'w') as f:
            f.write("Iteration\tBest_Mutation\tMedian_Error\n")
    
    recommendations = recommend_worst_n(model_path, evc_path, prediction_path, global_start, global_end, n=50)
    recommendations.rename(columns={'Rank':'Iteration','Mutation':'Best_Mutation','Predicted_Error':'Median_Error'},inplace=True)
    recommendations['Iteration'] = recommendations['Iteration'] + last_it
    
    try:
        with open(meta_tiling_path, 'a') as f: # 'a' = append mode
            for index, row in recommendations.iterrows():
                iteration_count = row['Iteration']
                best_mutation_name = row['Best_Mutation']
                best_median_error = row['Median_Error']
                f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
        return recommendations
    except Exception as e:
        print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")

real_meta_tiling_path = 'D:/Downloads/Files to work with/RUB/Meta/RUB_reverse_meta_tiling_path_actual_ss1.tsv'
for i in range(15):
    print(f'ITERATION: {i}')
    prediction, median_error = run_ridge(WT_SEQUENCE, TRAIN_FILE_PATH, EC_FILE_PATH, tiling_path_filename)
    prediction_path = f'D:/Downloads/Files to work with/RUB/Meta/RUB_reversed_meta_prediction_{i}.csv'
    #prediction['Error'] = prediction['Error'].mask(prediction['Class']=='test',np.nan) (in case of leaky data)
    #prediction['Activity'] = prediction['Activity'].mask(prediction['Class']=='test',np.nan) (in case of leaky data)
    prediction.to_csv(prediction_path, index=False)
    recommendations = run_reverse_meta(model_path, evc_path, prediction_path, global_start, global_end, meta_tiling_path)
    try:
        with open(real_meta_tiling_path, 'a') as f: # 'a' = append mode
            for index, row in recommendations.iterrows():
                iteration_count = row['Iteration']
                best_mutation_name = row['Best_Mutation']
                print(f'Best Mutation: {best_mutation_name}')
                best_median_error = median_error
                f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
    except Exception as e:
        print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")
    