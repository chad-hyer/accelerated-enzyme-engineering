import os
import sys
import pandas as pd
import numpy as np
import warnings

# --- Core Model and Helper Imports ---
# --- MODIFICATION 1: Import the OPTIMIZED class ---
# This assumes 'AugmentedMLDE_Class_optimized.py' is in the same directory or python path
from AugmentedMLDE_Class_optimized import AugmentedMLDEmodel

# --- MODIFICATION 2: Add new imports needed for pre-processing ---
from sklearn.preprocessing import StandardScaler
import re

class KillTaskError(Exception):
    print('Killing Task')

# Suppress warnings for cleaner parallel output
warnings.filterwarnings('ignore')

# --- Constants ---

# --- Make all input paths absolute ---
# Get the directory where this script (Tiling_parallelized_HPC.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"
TRAIN_FILE_PATH = 'CA_train.xlsx'
TEST_FILE_PATH = 'CA_test.xlsx'
EC_FILE_PATH = 'CA_single_mutant_matrix.csv'
tiling_path_filename = 'D:/Downloads/Files to work with/meta_model/reversed_meta_tiling_path.tsv'

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

# 6. Get predictions
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