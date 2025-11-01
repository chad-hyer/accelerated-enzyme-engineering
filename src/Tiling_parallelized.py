import os
import time
import sys
import traceback
import pandas as pd
import numpy as np
import warnings
from joblib import Parallel, delayed

# --- Core Model and Helper Imports ---
# Assumes AugmentedMLDE_Class_dataframe_input.py is in the same directory or python path
from AugmentedMLDE_Class_dataframe_input import AugmentedMLDEmodel
from chimerax_simulate import create_chimera_attribute_file as defattr
from Tiling_helper_functions import create_error_matrix

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- Constants ---
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"
TRAIN_FILE_PATH = 'D:/Documents/GitHub/accelerated-enzyme-engineering/src/CA_train.xlsx'
TEST_FILE_PATH = 'D:/Documents/GitHub/accelerated-enzyme-engineering/src/CA_test.xlsx'
EC_FILE_PATH = 'D:/Documents/GitHub/accelerated-enzyme-engineering/src/CA_single_mutant_matrix.csv'
OUT_DIR = 'D:/Documents/Tiling_Experiment_Parallel'

# --- Parallel Helper Function ---
# This function is unchanged. It already has the hard-coded
# model logic ('AugmentedEC', 'One Hot') that you want.
def process_what_if(mutation_name, train_main_df, test_main_df, wt_seq, ec_file):
    """
    Runs one "what-if" scenario in a parallel process.
    
    1. Finds the mutation in the test set.
    2. Adds it to a temp training set.
    3. Removes it from a temp test set.
    4. Trains the "best" model ('AugmentedEC', 'One Hot').
    5. Calculates the median error on the temp test set.
    """
    try:
        # 1. Find the row for the mutation we're testing
        # We must use .copy() to avoid race conditions and slicing warnings
        mutation_to_add_row = test_main_df[test_main_df['AminoAcid'] == mutation_name].copy()
        if mutation_to_add_row.empty:
            return (mutation_name, np.inf, None, "Mutation not found in test set")

        # 2. Create the temp training set
        temp_train_df = pd.concat([train_main_df, mutation_to_add_row], ignore_index=True)
        
        # 3. Create the temp test set (this is the key logic)
        temp_test_df = test_main_df.drop(mutation_to_add_row.index)
        
        # If the temp test set is empty (on the last run), we can't get an error
        if temp_test_df.empty:
            return (mutation_name, 0.0, None, "Final mutation") # Return 0 error

        # 4. Instantiate the model (using the DataFrames)
        model = AugmentedMLDEmodel(
            training_data=temp_train_df, 
            test_data=temp_test_df,          
            wt_sequence=wt_seq,
            EC_file=ec_file,
            Maestro_file=None, # Per your defaults
            ESM_file=None      # Per your defaults
        )
        # 5. Train *only* the best model
        # This is already hard-coded as you requested
        model.train_and_predict(model='AugmentedEC', encoding='One Hot')
        # 6. Get predictions
        all_preds_df = model._predictions_df
        if all_preds_df is None:
            return (mutation_name, np.inf, None, "Model training failed")
            
        # 7. Calculate error ONLY on the temp test set
        test_preds = pd.merge(temp_test_df, all_preds_df, left_on='AminoAcid', right_on='Mutation')
        test_preds['Error'] = np.abs(test_preds['Activity'] - test_preds['Prediction'])
        median_error = np.median(test_preds['Error'])
        
        # 8. Get params
        params = None
        
        # Return the prediction DataFrame for the best mutation
        return (mutation_name, median_error, all_preds_df, None)
    
    except Exception as e:
        # Capture errors from the parallel process
        tb_str = traceback.format_exc()
        return (mutation_name, np.inf, None, f"Exception: {e}\n{tb_str}")

# --- Main Tiling Loop ---
def main():
    
    true_start = time.perf_counter()
    
    # --- 1. Setup ---
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'ChimeraX_defattr'), exist_ok=True)

    # --- MODIFIED CHECKPOINTING LOGIC ---
    # We will now resume based on the 'perfect_tiling_path.tsv' file.
    tiling_path_filename = os.path.join(OUT_DIR, 'perfect_tiling_path.tsv')
    
    # Load original data first
    try:
        train_original = pd.read_excel(TRAIN_FILE_PATH)
        test_original = pd.read_excel(TEST_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Fatal Error: Could not find original input file. {e}")
        print(f"Looked for: {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        sys.exit(1)

    if os.path.exists(tiling_path_filename):
        print(f"Found existing tiling path file. Resuming from checkpoint...")
        try:
            tiling_path_df = pd.read_csv(tiling_path_filename, sep='\t')
            if tiling_path_df.empty:
                # File was created but no iterations finished
                train_main = train_original
                test_main = test_original
                iteration_count = 0
            else:
                # Get iteration_count and list of mutations already added
                iteration_count = tiling_path_df['Iteration'].max()
                chosen_mutations = tiling_path_df['Best_Mutation'].tolist()
                
                # Re-create the current state of train_main and test_main
                rows_to_move = test_original[test_original['AminoAcid'].isin(chosen_mutations)]
                
                train_main = pd.concat([train_original, rows_to_move], ignore_index=True)
                test_main = test_original.drop(rows_to_move.index)
            
            print(f"Resuming from Iteration {iteration_count + 1}")
        except Exception as e:
            print(f"Error loading {tiling_path_filename}: {e}. Starting from scratch.")
            train_main = train_original
            test_main = test_original
            iteration_count = 0
    else:
        print("No tiling path file found. Starting from scratch.")
        train_main = train_original
        test_main = test_original
        iteration_count = 0
    
    n_cores = max(1, os.cpu_count() - 1)
    print(f"--- Tiling Experiment Starting ---")
    print(f"Using {n_cores} parallel cores.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Initial training set size: {len(train_main)}")
    print(f"Initial test set size: {len(test_main)}")
    
    
    # --- Tiling path file setup ---
    if iteration_count == 0:
        # If starting from scratch, create the file and write the header
        try:
            with open(tiling_path_filename, 'w') as f:
                f.write("Iteration\tBest_Mutation\tMedian_Error\n")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not write to log file {tiling_path_filename}. {e}")
            sys.exit(1)
    
    # --- 2. Main Tiling Loop ---
    while len(test_main) > 0:
        
        iteration_count += 1
        iteration_start_time = time.perf_counter()
        iteration_dir = os.path.join(OUT_DIR, f'it_{iteration_count}')
        
        # --- MODIFICATION: Create subfolders for artifacts ---
        pred_dir = os.path.join(iteration_dir, 'predictions')
        matrix_dir = os.path.join(iteration_dir, 'error_matrices')
        what_if_dir = os.path.join(iteration_dir, 'what_if_reports')
        
        os.makedirs(iteration_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(what_if_dir, exist_ok=True)
        
        print(f"\n--- Iteration {iteration_count} / Remaining Test Set: {len(test_main)} ---")
        
        mutations_to_process = test_main['AminoAcid'].tolist()
        
        if not mutations_to_process:
            print("No more mutations to test. Tiling complete.")
            break

        print(f"Testing {len(mutations_to_process)} candidate mutations in parallel...")

        # --- C. Run "What-If" Scenarios (Parallel) ---
        parallel_start_time = time.perf_counter()
        
        results = Parallel(n_jobs=n_cores)(delayed(process_what_if)(
            mut_name,       
            train_main,     
            test_main,      
            WT_SEQUENCE,    
            EC_FILE_PATH    # Pass the absolute path
        ) for mut_name in mutations_to_process)
        
        print(f"Parallel 'what-if' finished in {time.perf_counter() - parallel_start_time:.2f}s")
        
        # --- D. Process Results and Find Best (Serial) ---
        it_dict = {}
        it_params = {}
        it_predictions = {}
        
        failed_jobs = 0
        for res in results:
            if res is None: 
                failed_jobs += 1
                continue
            mut_name, error, pred_df, params = res
            it_dict[mut_name] = error
            it_params[mut_name] = params
            it_predictions[mut_name] = pred_df
            if error == np.inf:
                failed_jobs += 1

        if failed_jobs > 0:
             print(f"Warning: {failed_jobs} parallel jobs failed (see errors above).")

        # Find the mutation that *minimized* the median error
        best_mutation_name = min(it_dict, key=it_dict.get)
        best_median_error = it_dict[best_mutation_name]
        best_prediction_df = it_predictions[best_mutation_name]
        
        if best_median_error == np.inf:
            print(f"Error: All {len(mutations_to_process)} parallel jobs failed. Stopping experiment.")
            print("Please check the error messages printed above.")
            break 

        print(f"Best mutation to add: {best_mutation_name} (New Median Error: {best_median_error:.4f})")
        
        # --- Append this iteration's result to the TSV file ---
        try:
            with open(tiling_path_filename, 'a') as f: # 'a' = append mode
                f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {tiling_path_filename}. {e}")
        
        # --- E. Commit Change for Next Loop (Serial) ---
        row_to_add = test_main[test_main['AminoAcid'] == best_mutation_name].copy()
        
        # Update the *main* dataframes for the next iteration
        train_main = pd.concat([train_main, row_to_add], ignore_index=True)
        test_main = test_main.drop(row_to_add.index)
        
        # --- F. Save Artifacts (Serial) ---
        # --- MODIFICATION: Removed temp file saving, as we now resume from originals ---
        
        # Save the *prediction CSV* from the best run
        pred_csv_path = os.path.join(pred_dir, f'prediction_it_{iteration_count}.csv')
        if best_prediction_df is not None:
            best_prediction_df.to_csv(pred_csv_path, index=False)
        
        # --- Save the sorted it_dict as a CSV ---
        it_dict_df = pd.DataFrame(it_dict.items(), columns=['Mutation', 'Median_Error'])
        it_dict_df.sort_values(by='Median_Error', ascending=True, inplace=True)
        it_dict_path = os.path.join(what_if_dir, 'all_what_if_errors.csv')
        it_dict_df.to_csv(it_dict_path, index=False)
        
        # --- Save Error Matrix and Defattr (only if predictions exist) ---
        if best_prediction_df is not None:
            best_prediction_df.rename(columns={'Mutation': 'AminoAcid'}, inplace=True)

            train_labels = pd.DataFrame({'AminoAcid': train_main['AminoAcid'], 'Class': 'train'})
            test_labels = pd.DataFrame({'AminoAcid': test_main['AminoAcid'], 'Class': 'test'})
            all_labels = pd.concat([train_labels, test_labels])
            
            # We need the full (train+test) activity data for merging
            # We can use the original dataframes for a complete activity lookup
            full_activity_data = pd.concat([train_original, test_original])[['AminoAcid', 'Activity']].drop_duplicates()
            
            matrix_input_df = pd.merge(best_prediction_df, all_labels, on='AminoAcid', how='left')
            matrix_input_df = pd.merge(matrix_input_df, full_activity_data, on='AminoAcid', how='left')
            
            matrix_input_df['Error'] = np.abs(matrix_input_df['Activity'] - matrix_input_df['Prediction'])
            
            error_matrix = create_error_matrix(matrix_input_df)
            # --- MODIFICATION: Save to new matrix_dir subfolder ---
            error_matrix.to_csv(os.path.join(matrix_dir, f'error_it_{iteration_count}.csv'))
            
            # Save ChimeraX attribute file
            defattr_path = os.path.join(OUT_DIR, 'ChimeraX_defattr', f'it_{iteration_count}.defattr')
            # --- MODIFICATION: Save to new pred_dir subfolder ---
            defattr_input_path = os.path.join(pred_dir, f'defattr_input_it_{iteration_count}.csv')
            matrix_input_df.to_csv(defattr_input_path, index=False)
            
            defattr(defattr_input_path, defattr_path, error_column_name='Error')

        print(f"Iteration {iteration_count} finished in {time.perf_counter() - iteration_start_time:.2f}s")
        print(f"Remaining test set size: {len(test_main)}")

    total_time = time.perf_counter() - true_start
    
    # --- No final save needed, just print location ---
    print(f"\nPerfect tiling path saved to {tiling_path_filename}")
    
    print(f"\n--- Tiling Experiment Complete ---")
    print(f"Total time: {total_time / 3600:.2f} hours")

# --- Run the main function ---
if __name__ == '__main__':
    main()

