# Tiling_HPC_Conductor.py
# (All your imports from the original script)
import subprocess # Add this
import glob       # Add this
import pickle     # Add this

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


# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- Constants ---

# --- Make all input paths absolute ---
# Get the directory where this script (Tiling_parallelized_HPC.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"
TRAIN_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/CA_train.xlsx'
TEST_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/CA_test.xlsx'
EC_FILE_PATH = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/CA_single_mutant_matrix.csv'
OUT_DIR = '/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/results'


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

def main():
    
    true_start = time.perf_counter()
    
    # --- 1. Setup ---
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'ChimeraX_defattr'), exist_ok=True)
    tiling_path_filename = os.path.join(OUT_DIR, 'perfect_tiling_path.tsv')
    
    # Load original data first
    try:
        train_original = pd.read_excel(TRAIN_FILE_PATH)
        test_original = pd.read_excel(TEST_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Fatal Error: Could not find original input file. {e}")
        sys.exit(1)
        
    iteration_count = 0 # Initialize

    # --- 2. Main Tiling Loop ---
    while True: 
        
        # --- A. Checkpoint Logic (Copied from original script) ---
        # This logic runs at the START of each loop to load the
        # *current* state of the experiment.
        if os.path.exists(tiling_path_filename):
            try:
                tiling_path_df = pd.read_csv(tiling_path_filename, sep='\t')
                if tiling_path_df.empty:
                    train_main = train_original.copy()
                    test_main = test_original.copy()
                    iteration_count = 0 # No iterations have run
                else:
                    iteration_count = tiling_path_df['Iteration'].max() # Get last completed iteration
                    chosen_mutations = tiling_path_df['Best_Mutation'].tolist()
                    rows_to_move = test_original[test_original['AminoAcid'].isin(chosen_mutations)]
                    train_main = pd.concat([train_original, rows_to_move], ignore_index=True)
                    test_main = test_original[~test_original['AminoAcid'].isin(chosen_mutations)].copy()
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
            # Write the header
            try:
                with open(tiling_path_filename, 'w') as f:
                    f.write("Iteration\tBest_Mutation\tMedian_Error\n")
            except Exception as e:
                print(f"CRITICAL ERROR: Could not write to log file {tiling_path_filename}. {e}")
                sys.exit(1)

        # --- B. Check for Completion ---
        if len(test_main) == 0:
            print("No more mutations to test. Tiling complete.")
            break
            
        # --- C. (THE FIX) Increment Iteration and Create Directories ---
        # We are now starting the *next* iteration.
        iteration_count += 1
        iteration_start_time = time.perf_counter()
        
        # Define and create the directory for *this* iteration's results
        iteration_dir = os.path.join(OUT_DIR, f'it_{iteration_count}')
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Define and create the log directory for *this* iteration's workers
        log_dir = os.path.join("/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src", "slurm_logs", f"it_{iteration_count}")
        os.makedirs(log_dir, exist_ok=True)
        # --- END OF FIX ---

        print(f"\n--- Iteration {iteration_count} / Remaining Test Set: {len(test_main)} ---")
        
        mutations_to_process = test_main['AminoAcid'].tolist()
        n_tasks = len(mutations_to_process)
        print(f"Preparing to test {n_tasks} mutations via Slurm job array...")

        # --- D. Prepare data for parallel workers ---
        # Now we can safely write the mutation list
        mutation_list_path = os.path.join(iteration_dir, 'mutation_list.txt')
        with open(mutation_list_path, 'w') as f:
            for mut in mutations_to_process: f.write(f"{mut}\n")
            
        # --- E. Run Tiled Error Scenarios ---
        parallel_start_time = time.perf_counter()
        
        # --- NEW CHUNKING LOGIC ---
        
        # Set a safe chunk size, based on your observation
        MAX_ARRAY_CHUNK_SIZE = 999 
        
        # Set a max number of *concurrent* jobs (e.g., 500).
        # This is polite to the scheduler and adds '%500' to the array spec.
        MAX_CONCURRENT_JOBS = 500

        # Calculate number of chunks needed
        num_chunks = (n_tasks + MAX_ARRAY_CHUNK_SIZE - 1) // MAX_ARRAY_CHUNK_SIZE

        if num_chunks > 1:
            print(f"Total tasks ({n_tasks}) exceeds chunk size ({MAX_ARRAY_CHUNK_SIZE}).")
            print(f"Submitting in {num_chunks} chunks.")
        else:
            print(f"Submitting {n_tasks} tasks in a single chunk.")

        # Loop over each chunk and submit it
        for i in range(num_chunks):
            # Calculate the start and end index for this chunk (1-based)
            task_start_index = (i * MAX_ARRAY_CHUNK_SIZE) + 1
            task_end_index = min((i + 1) * MAX_ARRAY_CHUNK_SIZE, n_tasks)
            
            # Create the array spec, e.g., "1-999%500" or "1000-1998%500"
            array_spec = f"{task_start_index}-{task_end_index}%{MAX_CONCURRENT_JOBS}"

            if num_chunks > 1:
                print(f"--- Submitting Chunk {i+1}/{num_chunks}: --array={array_spec} ---")

            sbatch_command = [
                'sbatch',
                '--wait', # CRITICAL: Wait for THIS CHUNK to finish
                f'--array={array_spec}', # Use the chunked and throttled spec
                f'--output={log_dir}/worker_%A_%a.out',
                f'--error={log_dir}/worker_%A_%a.err',
                'sbatch_array_worker.sbatch', 
                iteration_dir,                # Arg 1: IT_DIR
                mutation_list_path,           # Arg 2: MUTATION_LIST_PATH
            ]
            
            print(f"Submitting: {' '.join(sbatch_command)}")
            try:
                # Run the sbatch command for the current chunk
                subprocess.run(sbatch_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"FATAL ERROR: Slurm job array chunk {i+1} failed: {e}.")
                print(f"Command: {repr(e.cmd)}") 
                print(f"Stderr (if any): {e.stderr}")
                print("Check Slurm logs. Aborting experiment.")
                sys.exit(1)

        # --- END OF NEW CHUNKING LOGIC ---

        print(f"All {num_chunks} chunk(s) finished in {time.perf_counter() - parallel_start_time:.2f}s")
        
        # --- F. Process Results and Find Best (Serial) ---
        print("Collecting results from worker jobs...")
        it_dict = {}
        error_files = glob.glob(os.path.join(iteration_dir, 'error_*.csv'))
        
        if len(error_files) != n_tasks:
            print(f"Warning: Expected {n_tasks} result files, but found {len(error_files)}.")

        error_dfs = []
        for f in error_files:
            try:
                error_dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"Warning: could not read {f}. {e}")
        
        if not error_dfs:
            print(f"FATAL ERROR: No result files found for iteration {iteration_count}. Exiting.")
            sys.exit(1)
            
        full_error_df = pd.concat(error_dfs)
        it_dict = pd.Series(full_error_df.Median_Error.values, index=full_error_df.Mutation).to_dict()

        # Handle failed jobs
        if not it_dict:
             print(f"FATAL ERROR: All jobs failed for iteration {iteration_count}. Exiting.")
             sys.exit(1)

        best_mutation_name = min(it_dict, key=it_dict.get)
        best_median_error = it_dict[best_mutation_name]
        
        if best_median_error == np.inf:
            print(f"Error: All parallel jobs failed (best error is inf). Stopping experiment.")
            sys.exit(1)

        print(f"Best mutation to add: {best_mutation_name} (New Median Error: {best_median_error:.4f})")
        
        # Append this iteration's result to the TSV file
        try:
            with open(tiling_path_filename, 'a') as f: # 'a' = append mode
                f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {tiling_path_filename}. {e}")
        
        # --- G. Save Artifacts (Serial) ---
        # (This logic is from your original script)
        it_dict_df = pd.DataFrame(it_dict.items(), columns=['Mutation', 'Median_Error'])
        it_dict_df.sort_values(by='Median_Error', ascending=True, inplace=True)
        it_dict_path = os.path.join(iteration_dir, f'median_error_it_{iteration_count}.csv')
        it_dict_df.to_csv(it_dict_path, index=False)
        
        best_prediction_path = os.path.join(iteration_dir, f'predictions_{best_mutation_name}.pkl')
        best_prediction_df = None
        try:
            with open(best_prediction_path, 'rb') as f:
                best_prediction_df = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load best prediction file {best_prediction_path}. {e}")

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
            error_matrix.to_csv(os.path.join(iteration_dir, f'best_error_it_{iteration_count}.csv'))
            
            prediction_csv_path = os.path.join(iteration_dir, f'prediction_it_{iteration_count}.csv')
            matrix_input_df.to_csv(prediction_csv_path, index=False)
            
            defattr_path = os.path.join(OUT_DIR, 'ChimeraX_defattr', f'it_{iteration_count}.defattr')
            defattr(prediction_csv_path, defattr_path, error_column_name='Error')

        print(f"Iteration {iteration_count} finished in {time.perf_counter() - iteration_start_time:.2f}s")
        
        # --- H. Cleanup Intermediate Files ---
        print("Cleaning up intermediate files...")
        files_to_remove = glob.glob(os.path.join(iteration_dir, 'error_*.csv'))
        files_to_remove.extend(glob.glob(os.path.join(iteration_dir, 'predictions_*.pkl')))
        files_to_remove.append(mutation_list_path)
        
        for f in files_to_remove:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove file {f}. {e}")

    # --- Loop ends when break condition is met ---
    
    total_time = time.perf_counter() - true_start
    print(f"\nPerfect tiling path saved to {tiling_path_filename}")
    print(f"\n--- Tiling Experiment Complete ---")
    print(f"Total time: {total_time / 3600:.2f} hours")

# --- Run the main function ---
if __name__ == '__main__':
    main()