import pandas as pd
from AugmentedMLDE_Class import AugmentedMLDEmodel
# Make sure your modified 'AugmentedMLDE_HelperFuncs.py' is in the same folder

# --- 1. Define Your Parameters ---

# Your full wild-type protein sequence
# This is still required by the 'encode' function to build the feature vectors
WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK" 

# --- 2. Define File Paths ---
TRAIN_FILE = 'CA_train.xlsx'
TEST_FILE = 'CA_test.xlsx' # Use None if you don't have a separate test set

# Your EVCouplings file is now the "anchor" file for the model's scope
EC_FILE = 'CA_single_mutant_matrix.csv' 

# Other zero-shot predictor files
ESM_FILE = 'ESM_file.csv'
MAESTRO_FILE = 'Maestro_file.csv'

# --- 3. Instantiate and Run the Model ---
print("Initializing Augmented MLDE Model...")
print("Defining model scope from EVC file + Training file...")

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

# --- 4. Run Model Comparison ---
print("Running model comparison...")
# This will test all encodings and augmentations
model.compare_and_predict(ShowPlots=False)

# --- 5. Access Results ---
print("Model comparison complete.")

# Get the results DataFrame
model_metrics = model._model_metrics
print("Model Performance Metrics:")
print(model_metrics.sort_values('spearman_r', ascending=False))

# Save metrics to a CSV
model_metrics.to_csv('ssm_model_metrics.csv', index=False)

# --- 6. (Optional) Train and Predict with the Best Model ---
# Let's say 'AugmentedEC_ESM' with 'ZScales' was your best model
print("Training final model...")
model.train_and_predict(model='AugmentedEC_ESM', encoding='One Hot')

# Get the final predictions for the entire model scope
all_predictions_df = model._predictions_df
if all_predictions_df is not None:
    print("Final predictions for all mutations in scope (EVC + Train):")
    print(all_predictions_df.head())

    # Save all predictions
    all_predictions_df.to_csv('ssm_all_predictions.csv', index=False)
else:
    print("Skipping final prediction saving because training failed or was skipped.")

# Save all predictions
all_predictions_df.to_csv('ssm_all_predictions.csv', index=False)

print("Process finished.")