# process_one_mutation.py
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
import warnings
import re # Add this
from sklearn.preprocessing import StandardScaler # Add this

# --- Core Model and Helper Imports ---
from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- CONSTANTS ---
# (Copied from the main script)
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"
# --- ALL ORIGINAL FILE PATHS ARE NOW DEFINED HERE ---
BASE_DIR = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src'
TRAIN_FILE_PATH = os.path.join(BASE_DIR, 'CA_train.xlsx')
TEST_FILE_PATH = os.path.join(BASE_DIR, 'CA_test.xlsx')
EC_FILE_PATH = os.path.join(BASE_DIR, 'CA_single_mutant_matrix.csv')
TILING_PATH_FILE = os.path.join(BASE_DIR, 'results', 'perfect_tiling_path.tsv')

# --- HELPER FUNCTIONS ---
# (We must copy preprocess_evc_data here)
def preprocess_evc_data(ec_file_path, train_df, test_df):
    """
    Loads and processes the EVC data.
    """
    anchor_data = pd.read_csv(ec_file_path)
    anchor_data.rename(columns={'mutant': 'Mutations', 'prediction_epistatic': 'Predictions'}, inplace=True)
    evc_mutations_set = set(anchor_data['Mutations'])
    def is_wt(mutation_str):
        match = re.match(r"([A-Z])(\d+)([A-Z])", str(mutation_str))
        return match and match.group(1) == match.group(3)
    train_df_filtered = train_df[~train_df['AminoAcid'].apply(is_wt)]
    
    full_dataset_mutations = set(train_df['AminoAcid']).union(set(test_df['AminoAcid']))
    model_scope_set = evc_mutations_set.union(full_dataset_mutations)
    model_scope_mutations = sorted(list(model_scope_set))
    
    sc = StandardScaler()
    master_series = pd.Series(index=model_scope_mutations, dtype=float)
    raw_series = pd.Series(anchor_data['Predictions'].values, index=anchor_data['Mutations'])
    master_series.update(raw_series)

    non_nan_mask = master_series.notna()
    if non_nan_mask.any():
        scores_to_scale = np.array(master_series[non_nan_mask]).reshape(-1, 1)
        scaled_scores = sc.fit_transform(scores_to_scale)
        master_series[non_nan_mask] = scaled_scores.flatten()

    master_series.fillna(0.0, inplace=True)
    return master_series, model_scope_mutations

# (This is the original function, unchanged)
def process_tiled_error(mutation_name, train_main_df, test_main_df, 
                        wt_seq, model_scope_mutations, ec_predictions):
    try:
        mutation_to_add_row = test_main_df[test_main_df['AminoAcid'] == mutation_name].copy()
        if mutation_to_add_row.empty:
            return (mutation_name, np.inf, None, "Mutation not found in test set")
        
        temp_train_df = pd.concat([train_main_df, mutation_to_add_row], ignore_index=True)
        temp_test_df = test_main_df.drop(mutation_to_add_row.index)
        
        if temp_test_df.empty:
            return (mutation_name, 0.0, None, "Final mutation") 

        model = AugmentedMLDEmodel(
            training_data=temp_train_df, 
            test_data=temp_test_df,          
            wt_sequence=wt_seq,
            model_scope_mutations=model_scope_mutations,
            ec_predictions=ec_predictions,
            Maestro_file=None, 
            ESM_file=None      
        )
        model.train_and_predict(model='AugmentedEC', encoding='One Hot')
        all_preds_df = model._predictions_df
        if all_preds_df is None:
            return (mutation_name, np.inf, None, "Model training failed")
            
        test_preds = pd.merge(temp_test_df, all_preds_df, left_on='AminoAcid', right_on='Mutation')
        test_preds['Error'] = np.abs(test_preds['Activity'] - test_preds['Prediction'])
        median_error = np.median(test_preds['Error'])
        
        return (mutation_name, median_error, all_preds_df, "Success")
    
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"Exception: {e}\n{tb_str}"
        return (mutation_name, np.inf, None, error_msg)

# --- Main execution for the worker ---
if __name__ == "__main__":
    
    # 1. Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python process_one_mutation.py <it_dir> <mutation_list_path> <task_id>")
        sys.exit(1)
        
    IT_DIR = sys.argv[1]
    MUTATION_LIST_PATH = sys.argv[2]
    TASK_ID = int(sys.argv[3]) # This is the SLURM_ARRAY_TASK_ID

    MUTATION_NAME = ""
    try:
        # 2. Get the mutation name from the list
        with open(MUTATION_LIST_PATH, 'r') as f:
            mutations = f.read().splitlines()
            if 0 < TASK_ID <= len(mutations):
                MUTATION_NAME = mutations[TASK_ID - 1]
            else:
                raise IndexError(f"Task ID {TASK_ID} is out of bounds for mutation list (size {len(mutations)}).")
        if not MUTATION_NAME:
             raise ValueError("Mutation name is empty.")

        # 3. --- NEW: Load all original data ---
        train_original = pd.read_excel(TRAIN_FILE_PATH)
        test_original = pd.read_excel(TEST_FILE_PATH)
        
        # 4. --- NEW: Re-build state from perfect_tiling_path.tsv ---
        if os.path.exists(TILING_PATH_FILE):
            tiling_path_df = pd.read_csv(TILING_PATH_FILE, sep='\t')
            if tiling_path_df.empty:
                train_main = train_original.copy()
                test_main = test_original.copy()
            else:
                chosen_mutations = tiling_path_df['Best_Mutation'].tolist()
                rows_to_move = test_original[test_original['AminoAcid'].isin(chosen_mutations)]
                train_main = pd.concat([train_original, rows_to_move], ignore_index=True)
                test_main = test_original[~test_original['AminoAcid'].isin(chosen_mutations)].copy()
        else:
            # This case happens on iteration 1
            train_main = train_original.copy()
            test_main = test_original.copy()
            
        # 5. --- NEW: Run pre-processing ---
        ec_data, model_scope = preprocess_evc_data(EC_FILE_PATH, train_original, test_original)
        # Note: We pass train_original and test_original to get the FULL model scope.

    except Exception as e:
        print(f"Error during setup: {e}")
        error_df = pd.DataFrame([{'Mutation': f"TASK_{TASK_ID}_FAILED", 'Median_Error': np.inf}])
        error_file = os.path.join(IT_DIR, f"error_TASK_{TASK_ID}.csv")
        error_df.to_csv(error_file, index=False)
        sys.exit(1)

    # 6. Run the single job
    mut_name, error, pred_df, status = process_tiled_error(
        MUTATION_NAME,
        train_main,
        test_main,
        WT_SEQUENCE,
        model_scope,
        ec_data
    )
    
    # 7. Save the results to unique files
    error_df = pd.DataFrame([{'Mutation': mut_name, 'Median_Error': error}])
    error_file = os.path.join(IT_DIR, f"error_{mut_name}.csv")
    error_df.to_csv(error_file, index=False)
    
    if pred_df is not None:
        pred_file = os.path.join(IT_DIR, f"predictions_{mut_name}.pkl")
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_df, f)

    if status != "Success":
        print(f"Job {mut_name} failed: {status}")
    else:
        print(f"Job {mut_name} succeeded. Error: {error}")