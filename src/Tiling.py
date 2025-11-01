import os, time, sys, traceback
import pandas as pd
import numpy as np
from AugmentedMLDE_Class import AugmentedMLDEmodel
from chimerax_simulate import create_chimera_attribute_file as defattr
from Tiling_helper_functions import *

import warnings
warnings.filterwarnings('ignore')

true_start = time.perf_counter()
# --- 1. Define Your Parameters ---
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK" 

# --- 2. Define File Paths ---
TRAIN_FILE = 'CA_train.xlsx'
TEST_FILE = 'CA_test.xlsx' # Use None if you don't have a separate test set
out_dir = 'D:/Documents/Tiling_Experiment'

# Your EVCouplings file is now the "anchor" file for the model's scope
EC_FILE = 'CA_single_mutant_matrix.csv' 

# --- 3. Instantiate and Run the Model ---
print("Initializing Augmented MLDE Model...")
print("Defining model scope from EVC file + Training file...")

start = time.perf_counter()
# Instantiate the class with the new signature
# Notice 'sequence_length' is no longer passed
model = AugmentedMLDEmodel(
    training_data_file=TRAIN_FILE,
    test_data_file=TEST_FILE,
    wt_sequence=WT_SEQUENCE,
    EC_file=EC_FILE,
    Maestro_file=None,
    ESM_file=None
)

model.train_and_predict(model='AugmentedEC', encoding='One Hot')
prediction = model._predictions_df
prediction.rename(columns={'Mutation':'AminoAcid'},inplace=True)
prediction.dropna(inplace=True)
train = pd.read_excel(TRAIN_FILE)
test = pd.read_excel(TEST_FILE)
train['Class'] = 'train'
test['Class'] = 'test'

data = train._append(test, ignore_index=True)
prediction = prediction.merge(data, how='left', on='AminoAcid')
prediction['Error'] = abs((prediction['Activity'] - prediction['Prediction']) / prediction['Activity'])
prediction['Original'] = prediction['AminoAcid'].str[0]
prediction['Mutation'] = prediction['AminoAcid'].str[-1]
prediction['Residue Number'] = prediction['AminoAcid'].str[1:-1].astype(int)
prediction['logError'] = np.log(prediction['Error'] + 1e-5)
prediction.to_csv(f'{out_dir}/prediction_it_0.csv', index=False)
error_matrix = create_error_matrix(prediction)
error_matrix.to_csv(f'{out_dir}/error_it_0.csv')
defattr(f'{out_dir}/prediction_it_0.csv', f'{out_dir}/ChimeraX_defattr/it_0.defattr')
end = time.perf_counter()
elapsed = end - start

iteration = 0
full_params = pd.DataFrame(columns=['Iteration','AminoAcid','Median Error'])
while iteration < 5: #len(train) < len(data):
    start = time.perf_counter()
    iteration +=1
    os.makedirs(f'it_{iteration}', exist_ok=True)
    print(f'Testing all possible training data combinations for iteration {iteration}. Last iteration took {elapsed:.6f} seconds')
    it_params = pd.DataFrame(columns=['AminoAcid','Median Error'])
    it_dict = {}
    for index, row in data.iterrows():
        mut = row['AminoAcid']
        if mut not in train:
            try:
                train_it = train._append(row, ignore_index=True)
                test_it = test[test['AminoAcid']!=mut]
                train_it.to_excel(f'{out_dir}/TRAIN_TMP.xlsx', index=False)
                test_it.to_excel(f'{out_dir}/TEST_TMP.xlsx', index=False)
                model = AugmentedMLDEmodel(
                    training_data_file=f'{out_dir}/TRAIN_TMP.xlsx',
                    test_data_file=f'{out_dir}/TEST_TMP.xlsx',
                    wt_sequence=WT_SEQUENCE,
                    EC_file=EC_FILE,
                    Maestro_file=None,
                    ESM_file=None
                )

                model.train_and_predict(model='AugmentedEC', encoding='One Hot')
                prediction_it = model._predictions_df
                prediction_it.rename(columns={'Mutation':'AminoAcid'},inplace=True)
                prediction_it.dropna(inplace=True)
                train_it['Class'] = 'train'
                test_it['Class'] = 'test'

                data_it = train_it._append(test_it, ignore_index=True)
                prediction_it = prediction_it.merge(data_it, how='left', on='AminoAcid')
                prediction_it['Error'] = abs((prediction_it['Activity'] - prediction_it['Prediction']) / prediction_it['Activity'])
                prediction_it['Original'] = prediction_it['AminoAcid'].str[0]
                prediction_it['Mutation'] = prediction_it['AminoAcid'].str[-1]
                prediction_it['Residue Number'] = prediction_it['AminoAcid'].str[1:-1].astype(int)
                prediction_it['logError'] = np.log(prediction_it['Error'] + 1e-5)
                median_error = prediction_it[prediction_it['Class']=='test']['Error'].median()
                it_params = it_params._append(pd.Series({'AminoAcid':mut,'Median Error':median_error}), ignore_index=True)
                it_dict.update({mut:prediction_it})
            except Exception:
                print(f"Error with {mut}:")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"Exception Type: {exc_type}")
                print(f"Exception Value: {exc_value}")
                print("Traceback:")
                traceback.print_tb(exc_traceback)
        
    ### CHOOSE BEST ERROR ###
    it_params.sort_values(by='Median Error', inplace=True)
    it_params.reset_index(drop=True, inplace=True)
    print(it_params.head())
    best = it_params.loc[0]
    mut = best['AminoAcid']
    prediction = it_dict[mut]
    median_error = best['Median Error']
    print(f'Adding {mut} to training set')
    train = train._append(best, ignore_index=True)
    test = test[test['Amino Acid']!=mut]
    train.to_excel(f'{out_dir}/TRAIN_TMP.xlsx', index=False)
    test.to_excel(f'{out_dir}/TEST_TMP.xlsx', index=False)
    prediction.to_csv(f'{out_dir}/it_{iteration}/prediction_it_{iteration}.csv', index=False)
    error_matrix = create_error_matrix(prediction)
    error_matrix.to_csv(f'{out_dir}/it_{iteration}/error_it_{iteration}.csv')
    defattr(f'{out_dir}/it_{iteration}/prediction_it_{iteration}.csv', f'{out_dir}/ChimeraX_defattr/it_{iteration}.defattr')
    full_params = full_params._append(pd.Series({'Iteration':iteration,'AminoAcid':mut,'Median Error':median_error}), ignore_index=True)
    end = time.perf_counter()
    elapsed = end - start

full_params.to_csv(f'{out_dir}/full_params.csv', index=False)
true_end = time.perf_counter()
true_elapsed = true_end - true_start
print(f'Workflow complete and took {elapsed:.6f} seconds')