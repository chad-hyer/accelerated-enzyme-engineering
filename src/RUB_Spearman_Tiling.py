import os
import time
import sys
import traceback
import pandas as pd
import numpy as np
import warnings
from joblib import Parallel, delayed

# --- Core Model and Helper Imports ---
# --- MODIFICATION 1: Import the OPTIMIZED class ---
# This assumes 'AugmentedMLDE_Class_optimized.py' is in the same directory or python path
from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel
from chimerax_simulate import create_chimera_attribute_file as defattr
from Tiling_helper_functions import create_error_matrix

# --- MODIFICATION 2: Add new imports needed for pre-processing ---
from sklearn.preprocessing import StandardScaler
import re
import scipy.stats as stats

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- Constants ---

# --- Make all input paths absolute ---
# Get the directory where this script (Tiling_parallelized_HPC.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
WT_SEQUENCE = "MDQSSRYVNLALKEEDLIAGGEHVLCAYIMKPKAGYGYVATAAHFAAESSTGTNVEVCTTDDFTRGVDALVYEVDEARELTKIAYPVALFDRNITDGKAMIASFLTLTMGNNQGMGDVEYAKMHDFYVPEAYRALFDGPSVNISALWKVLGRPEVDGGLVVGTIIKPKLGLRPKPFAEACHAFWLGGDFIKNDEPQGNQPFAPLRDTIALVADAMRRAQDETGEAKLFSANITADDPFEIIARGEYVLETFGENASHVALLVDGYVAGAAAITTARRRFPDNFLHYHRAGHGAVTSPQSKRGYTAFVHCKMARLQGASGIHTGTMGFGKMEGESSDRAIAYMLTQDEAQGPFYRQSWGGMKACTPIISGGMNALRMPGFFENLGNANVILTAGGGAFGHIDGPVAGARSLRQAWQAWRDGVPVLDYAREHKELARAFESFPGDADQIYPGWRKALGVEDTRSALPA"
TRAIN_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/RUB/RUB_train.xlsx'
TEST_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/RUB/RUB_test.xlsx'
EC_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/RUB/RUB_single_mutant_matrix.csv'
OUT_DIR = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/RUB/spear'


# --- MODIFICATION 3: Add the EVC pre-processing function ---
def preprocess_evc_data(ec_file_path, train_df, test_df):
    """
    Loads and processes the EVC data ONCE.
    This is called at the start of main() before any loops.
    """
    print("Pre-processing EVC data once...")
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
    
    print("EVC data pre-processing complete.")
    return master_series, model_scope_mutations


# --- Parallel Helper Function ---
# --- MODIFICATION 4: Renamed to `process_tiled_error` and changed signature ---
def process_tiled_error(mutation_name, train_main_df, test_main_df, 
                        wt_seq, model_scope_mutations, ec_predictions):
    """
    Runs one "tiled error" scenario in a parallel process.
    """
    try:
        # 1. Find the row for the mutation we're testing
        mutation_to_add_row = test_main_df[test_main_df['AminoAcid'] == mutation_name].copy()
        if mutation_to_add_row.empty:
            return (mutation_name, np.inf, None, "Mutation not found in test set")

        # 2. Create the temp training set
        temp_train_df = pd.concat([train_main_df, mutation_to_add_row], ignore_index=True)
        
        # 3. Create the temp test set
        temp_test_df = test_main_df.drop(mutation_to_add_row.index)
        
        if temp_test_df.empty:
            return (mutation_name, 0.0, None, "Final mutation") 

        # 4. Instantiate the model (using the OPTIMIZED class)
        # --- MODIFICATION 5: Call the new optimized __init__ ---
        model = AugmentedMLDEmodel(
            training_data=temp_train_df, 
            test_data=temp_test_df,          
            wt_sequence=wt_seq,
            model_scope_mutations=model_scope_mutations, # Pass pre-processed
            ec_predictions=ec_predictions,               # Pass pre-processed
            Maestro_file=None, 
            ESM_file=None      
        )

        # 5. Train *only* the best model
        model.train_and_predict(model='AugmentedEC', encoding='One Hot')
        
        # 6. Get predictions
        all_preds_df = model._predictions_df
        if all_preds_df is None:
            return (mutation_name, np.inf, None, "Model training failed (all_preds_df is None)")
            
        # 7. Calculate error ONLY on the temp test set
        test_preds = pd.merge(temp_test_df, all_preds_df, left_on='AminoAcid', right_on='Mutation')
        test_preds['Error'] = np.abs(test_preds['Activity'] - test_preds['Prediction'])
        median_error = np.median(test_preds['Error'])
        spear, p = stats.spearmanr(test_preds['Prediction'],test_preds['Activity'])
        
        # 8. Get params
        params = model._best_alpha # This is set by the optimized class
        
        return (mutation_name, median_error, spear, all_preds_df, params)
    
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"Exception: {e}\n{tb_str}"
        print(f"--- ERROR in parallel job {mutation_name} ---\n{error_msg}\n--------------------")
        return (mutation_name, np.inf, None, error_msg)

# --- Main Tiling Loop ---
def main():
    
    true_start = time.perf_counter()
    
    # --- 1. Setup ---
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'ChimeraX_defattr'), exist_ok=True)

    tiling_path_filename = os.path.join(OUT_DIR, 'spear_tiling_path.tsv')
    
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
        print(f"Found existing tiling path file. Resuming from checkpoint...")
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
            
            print(f"Resuming from Iteration {iteration_count + 1}")
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
            print(f"Running on SLURM. Using {n_cores} allocated cores.")
        else:
            n_cores = max(1, os.cpu_count() - 1)
            print(f"Not on SLURM. Using {n_cores} local cores.")
    except Exception:
        n_cores = max(1, os.cpu_count() - 1)
        print(f"Error reading SLURM_CPUS_PER_TASK. Defaulting to {n_cores} local cores.")
    
    print(f"--- Tiling Experiment Starting ---")
    print(f"Initial training set size: {len(train_main)}")
    print(f"Initial test set size: {len(test_main)}")
    
    if iteration_count == 0:
        try:
            with open(tiling_path_filename, 'w') as f:
                f.write("Iteration\tBest_Mutation\tMedian_Error\tSpear\n")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not write to log file {tiling_path_filename}. {e}")
            sys.exit(1)
    
    # --- 2. Main Tiling Loop ---
    while len(test_main) > 0:
        
        iteration_count += 1
        iteration_start_time = time.perf_counter()
        iteration_dir = os.path.join(OUT_DIR, f'it_{iteration_count}')
        
        # --- MODIFICATION: Simplified folder structure ---
        os.makedirs(iteration_dir, exist_ok=True)
        
        print(f"\n--- Iteration {iteration_count} / Remaining Test Set: {len(test_main)} ---")
        
        mutations_to_process = test_main['AminoAcid'].tolist()
        
        if not mutations_to_process:
            print("No more mutations to test. Tiling complete.")
            break

        print(f"Testing {len(mutations_to_process)} candidate mutations in parallel...")

        # --- C. Run Tiled Error Scenarios (Parallel) ---
        parallel_start_time = time.perf_counter()
        
        # --- MODIFICATION 7: Renamed function and pass pre-processed data ---
        results = Parallel(n_jobs=n_cores)(delayed(process_tiled_error)(
            mut_name,       
            train_main,     
            test_main,      
            WT_SEQUENCE,    
            model_scope,           # Pass pre-processed data
            preprocessed_ec_data   # Pass pre-processed data
        ) for mut_name in mutations_to_process)
        
        print(f"Parallel processing finished in {time.perf_counter() - parallel_start_time:.2f}s")
        
        # --- D. Process Results and Find Best (Serial) ---
        it_dict = {}
        it_params = {}
        it_predictions = {}
        it_spear = {}

        failed_jobs = 0
        for res in results:
            if res is None: 
                failed_jobs += 1
                continue
            mut_name, error, spear, pred_df, params = res
            it_dict[mut_name] = error
            it_spear[mut_name] = spear
            it_params[mut_name] = params
            it_predictions[mut_name] = pred_df
            if error == np.inf:
                failed_jobs += 1

        if failed_jobs > 0:
             print(f"Warning: {failed_jobs} parallel jobs failed (see errors above).")

        best_mutation_name = max(it_spear, key=it_spear.get)
        best_median_error = it_dict[best_mutation_name]
        best_spear = it_spear[best_mutation_name]
        best_prediction_df = it_predictions[best_mutation_name]
        
        if best_median_error == np.inf:
            print(f"Error: All {len(mutations_to_process)} parallel jobs failed. Stopping experiment.")
            break 

        print(f"Best mutation to add: {best_mutation_name} (New Median Error: {best_median_error:.4f})")
        
        # Append this iteration's result to the TSV file
        try:
            with open(tiling_path_filename, 'a') as f: # 'a' = append mode
                f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\t{best_spear}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {tiling_path_filename}. {e}")
        
        # --- E. Commit Change for Next Loop (Serial) ---
        row_to_add = test_main[test_main['AminoAcid'] == best_mutation_name].copy()
        
        train_main = pd.concat([train_main, row_to_add], ignore_index=True)
        test_main = test_main[test_main['AminoAcid'] != best_mutation_name].copy()
        
        # --- F. Save Artifacts (Serial) ---
        
        # --- MODIFICATION: Save the sorted full error list ---
        it_dict_df = pd.DataFrame(it_dict.items(), columns=['Mutation', 'Median_Error'])
        it_dict_df.sort_values(by='Median_Error', ascending=True, inplace=True)
        it_dict_path = os.path.join(iteration_dir, f'median_error_it_{iteration_count}.csv')
        it_dict_df.to_csv(it_dict_path, index=False)
        
        # --- Save Error Matrix and Defattr (only if predictions exist) ---
        if best_prediction_df is not None:
            best_prediction_df.rename(columns={'Mutation': 'AminoAcid'}, inplace=True)

            train_labels = pd.DataFrame({'AminoAcid': train_main['AminoAcid'], 'Class': 'train'})
            test_labels = pd.DataFrame({'AminoAcid': test_main['AminoAcid'], 'Class': 'test'})
            all_labels = pd.concat([train_labels, test_labels])
            
            full_activity_data = pd.concat([train_original, test_original])[['AminoAcid', 'Activity']].drop_duplicates()
            
            matrix_input_df = pd.merge(best_prediction_df, all_labels, on='AminoAcid', how='left')
            matrix_input_df = pd.merge(matrix_input_df, full_activity_data, on='AminoAcid', how='left')
            
            matrix_input_df['Error'] = np.abs(matrix_input_df['Activity'] - matrix_input_df['Prediction'])
            
            error_matrix = create_error_matrix(matrix_input_df)
            # --- MODIFICATION: Save best error matrix ---
            error_matrix.to_csv(os.path.join(iteration_dir, f'best_error_it_{iteration_count}.csv'))
            
            # --- MODIFICATION: Save prediction data (formerly defattr_input) ---
            prediction_csv_path = os.path.join(iteration_dir, f'prediction_it_{iteration_count}.csv')
            matrix_input_df.to_csv(prediction_csv_path, index=False)
            
            # Save ChimeraX attribute file
            defattr_path = os.path.join(OUT_DIR, 'ChimeraX_defattr', f'it_{iteration_count}.defattr')
            defattr(prediction_csv_path, defattr_path, error_column_name='Error')

        print(f"Iteration {iteration_count} finished in {time.perf_counter() - iteration_start_time:.2f}s")
        print(f"Remaining test set size: {len(test_main)}")

    total_time = time.perf_counter() - true_start
    
    print(f"\nPerfect tiling path saved to {tiling_path_filename}")
    
    print(f"\n--- Tiling Experiment Complete ---")
    print(f"Total time: {total_time / 3600:.2f} hours")

# --- Run the main function ---
if __name__ == '__main__':
    main()