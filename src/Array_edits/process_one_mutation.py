# process_one_mutation.py
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
import warnings

# --- Core Model and Helper Imports ---
# These scripts must be in the same directory or in your PYTHONPATH
from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- Constants ---
# The WT_SEQUENCE must be defined here, as the worker needs it
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"

def process_tiled_error(mutation_name, train_main_df, test_main_df, 
                        wt_seq, model_scope_mutations, ec_predictions):
    """
    This is the exact function copied from your Tiling_parallelized_HPC_optimized.py
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
        
        return (mutation_name, median_error, all_preds_df, "Success")
    
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"Exception: {e}\n{tb_str}"
        print(f"--- ERROR in parallel job {mutation_name} ---\n{error_msg}\n--------------------")
        return (mutation_name, np.inf, None, error_msg)

# --- Main execution for the worker ---
if __name__ == "__main__":
    
    # 1. Parse command-line arguments
    if len(sys.argv) != 7:
        print("Usage: python process_one_mutation.py <it_dir> <mutation> <train_path> <test_path> <ec_path> <scope_path>")
        sys.exit(1)
        
    IT_DIR = sys.argv[1]
    MUTATION_NAME = sys.argv[2]
    TRAIN_PATH = sys.argv[3]
    TEST_PATH = sys.argv[4]
    EC_PATH = sys.argv[5]
    MODEL_SCOPE_PATH = sys.argv[6]

    # 2. Load the data from the temporary pickle files
    try:
        with open(TRAIN_PATH, 'rb') as f: train_main = pickle.load(f)
        with open(TEST_PATH, 'rb') as f: test_main = pickle.load(f)
        with open(EC_PATH, 'rb') as f: ec_data = pickle.load(f)
        with open(MODEL_SCOPE_PATH, 'rb') as f: model_scope = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        # Write a dummy error file so the conductor script knows this job failed
        error_df = pd.DataFrame([{'Mutation': MUTATION_NAME, 'Median_Error': np.inf}])
        error_file = os.path.join(IT_DIR, f"error_{MUTATION_NAME}.csv")
        error_df.to_csv(error_file, index=False)
        sys.exit(1)

    # 3. Run the single job
    mut_name, error, pred_df, status = process_tiled_error(
        MUTATION_NAME,
        train_main,
        test_main,
        WT_SEQUENCE,
        model_scope,
        ec_data
    )
    
    # 4. Save the results to unique files
    
    # File 1: The simple error score
    error_df = pd.DataFrame([{'Mutation': mut_name, 'Median_Error': error}])
    error_file = os.path.join(IT_DIR, f"error_{mut_name}.csv")
    error_df.to_csv(error_file, index=False)
    
    # File 2: The full prediction DataFrame
    if pred_df is not None:
        pred_file = os.path.join(IT_DIR, f"predictions_{mut_name}.pkl")
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_df, f)

    if status != "Success":
        print(f"Job {mut_name} failed: {status}")
    else:
        print(f"Job {mut_name} succeeded. Error: {error}")