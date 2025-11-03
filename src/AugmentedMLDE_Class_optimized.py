# -*- coding: utf-8 -*-
"""
This is a MODIFIED version of your class.

The __init__ method has been changed to accept:
1. `training_data`, `test_data` (as DataFrames)
2. `model_scope_mutations` (a pre-built list)
3. `ec_predictions` (a pre-processed pandas Series)

This AVOIDS all file I/O and pre-processing inside the
parallel loop, which is the main bottleneck.
"""

from AugmentedMLDE_HelperFuncs import encode, normalize_data, ALL_AAS
import pandas as pd
import numpy as np
from itertools import product
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import re, os

import time
from joblib import Parallel, delayed

# We use RidgeCV instead of GridSearchCV for massive speedup
from sklearn.linear_model import RidgeCV

model_list = ['Simple', 'AugmentedESM', 'AugmentedEC', 'AugmentedEnergy', 'AugmentedEC_Energy', 'AugmentedESM_Energy', 'AugmentedEC_ESM', 'AugmentedEC_Energy_ESM']
encoding_list = ['One Hot', 'Georgiev', 'ZScales', 'VHSE', 'Physical Descriptors'] 

class AugmentedMLDEmodel():
    
    # --- MODIFICATION 1: Changed __init__ signature ---
    # It now accepts pre-processed EVC data
    def __init__(self, training_data, test_data, wt_sequence, 
                 model_scope_mutations, ec_predictions, 
                 Maestro_file, ESM_file):
        
        # We no longer need the EC_file path
        # self._EC_file = EC_file 
        
        self._Maestro_file = Maestro_file
        self._ESM_file = ESM_file
        self._wt_sequence = wt_sequence
        self._data_normalization_type = 'Standardization'
        self._random_seed = 42
        self._all_predictions = None
        self._test_predictions = None
        self._model_metrics = None
        self._predictions_df = None
        self._best_alpha = None 

        # --- MODIFICATION 2: Use pre-processed data directly ---

        # 1. Use the pre-built master list
        self._model_scope_mutations = model_scope_mutations
        self._model_scope_set = set(model_scope_mutations)
        
        # 2. Use the provided training dataframe
        training_data_df = training_data.copy() 
        
        # Filter out WT from training data
        def is_wt(mutation_str):
            match = re.match(r"([A-Z])(\d+)([A-Z])", str(mutation_str))
            return match and match.group(1) == match.group(3)
        training_data_df = training_data_df[~training_data_df['AminoAcid'].apply(is_wt)]
        
        self._training_data = training_data_df

        # 3. Use the provided test dataframe
        if test_data is not None:
            test_data_df = test_data.copy() 
            test_data_df = test_data_df[test_data_df['AminoAcid'].isin(self._model_scope_set)]
            self._test_data = test_data_df
        else:
            self._test_data = None
        
        # 4. Use the pre-processed EVC data directly
        self._EC_predictions = ec_predictions

        # --- End of MODIFICATION 2 ---
        
        # This helper function is now ONLY for Maestro/ESM
        def prep_data_optional(file_loc):
            """
            Loads optional zero-shot files (Maestro/ESM), aligns, and fills.
            """
            sc = StandardScaler()
            raw = pd.read_csv(file_loc)
            
            # This assumes Maestro/ESM files also have 'mutant' and 'prediction_epistatic'
            raw.rename(columns={'mutant': 'Mutations', 'prediction_epistatic': 'Predictions'}, inplace=True)
            
            # Use the master list from __init__
            master_series = pd.Series(index=self._model_scope_mutations, dtype=float)
            raw_series = pd.Series(raw['Predictions'].values, index=raw['Mutations'])
            master_series.update(raw_series)

            non_nan_mask = master_series.notna()
            if non_nan_mask.any():
                scores_to_scale = np.array(master_series[non_nan_mask]).reshape(-1, 1)
                scaled_scores = sc.fit_transform(scores_to_scale)
                master_series[non_nan_mask] = scaled_scores.flatten()

            master_series.fillna(0.0, inplace=True)
            return master_series
        
        # We only run prep_data_optional for the files that exist
        if self._Maestro_file != None:
            self._Energy_predictions = prep_data_optional(self._Maestro_file)
        if self._ESM_file != None:
            self._ESM_predictions = prep_data_optional(self._ESM_file)
          
   
    def compare_and_predict(self, ShowPlots = False):
        # This function is unchanged from your version
        print("Starting parallel compare_and_predict...")
        t0 = time.time()
        
        encodings = {}
        for encoding_name in encoding_list:
            encodings[encoding_name] = {
                'x_train': encode(self._training_data, self._wt_sequence, self._model_scope_mutations)[encoding_name],
                'x_test': encode(self._test_data, self._wt_sequence, self._model_scope_mutations)[encoding_name],
                'x_all': encode(pd.DataFrame(), self._wt_sequence, self._model_scope_mutations)[encoding_name]
            }
        
        y_train = normalize_data(self._training_data, self._data_normalization_type)
        y_actual = normalize_data(self._test_data, self._data_normalization_type)
        
        has_ec = hasattr(self, '_EC_predictions')
        has_energy = hasattr(self, '_Energy_predictions')
        has_esm = hasattr(self, '_ESM_predictions')

        zero_shot_data = {}
        if has_ec:
            zero_shot_data['ec'] = {
                'train': np.array([[self._EC_predictions.loc[aa] for aa in self._training_data['AminoAcid']]]),
                'test': np.array([[self._EC_predictions.loc[aa] for aa in self._test_data['AminoAcid']]]),
                'all': np.array([self._EC_predictions[mut] for mut in self._model_scope_mutations]) # Aligned
            }
        if has_energy:
             zero_shot_data['energy'] = {
                'train': np.array([[self._Energy_predictions.loc[aa] for aa in self._training_data['AminoAcid']]]),
                'test': np.array([[self._Energy_predictions.loc[aa] for aa in self._test_data['AminoAcid']]]),
                'all': np.array([self._Energy_predictions[mut] for mut in self._model_scope_mutations]) # Aligned
            }
        if has_esm:
             zero_shot_data['esm'] = {
                'train': np.array([[self._ESM_predictions.loc[aa] for aa in self._training_data['AminoAcid']]]),
                'test': np.array([[self._ESM_predictions.loc[aa] for aa in self._test_data['AminoAcid']]]),
                'all': np.array([self._ESM_predictions[mut] for mut in self._model_scope_mutations]) # Aligned
            }

        def _fit_one_model(model_encoding_tuple):
            model, encoding = model_encoding_tuple
            
            if model == 'AugmentedEC' and not has_ec: return None
            if model == 'AugmentedEnergy' and not has_energy: return None
            if model == 'AugmentedESM' and not has_esm: return None
            if model == 'AugmentedEC_Energy' and not (has_ec and has_energy): return None
            if model == 'AugmentedEC_ESM' and not (has_ec and has_esm): return None
            if model == 'AugmentedESM_Energy' and not (has_esm and has_energy): return None
            if model == 'AugmentedEC_Energy_ESM' and not (has_ec and has_energy and has_esm): return None

            x_train = encodings[encoding]['x_train']
            x_test = encodings[encoding]['x_test']
            x_all = encodings[encoding]['x_all']
            
            # Build the augmented model inputs
            x_train_model, x_test_model, x_all_model = x_train, x_test, x_all
            
            if model == 'Simple':
                pass # Already set
            elif model == 'AugmentedEC':
                x_train_model = np.concatenate((zero_shot_data['ec']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['ec']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['ec']['all'].T,x_all), axis=1)
            elif model == 'AugmentedEnergy':
                x_train_model = np.concatenate((zero_shot_data['energy']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['energy']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['energy']['all'].T,x_all), axis=1)
            elif model == 'AugmentedESM':
                x_train_model = np.concatenate((zero_shot_data['esm']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['esm']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['esm']['all'].T,x_all), axis=1)
            elif model == 'AugmentedEC_Energy':
                x_train_model = np.concatenate((zero_shot_data['ec']['train'].T,zero_shot_data['energy']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['ec']['test'].T,zero_shot_data['energy']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['ec']['all'].T,zero_shot_data['energy']['all'].T,x_all), axis=1)
            elif model == 'AugmentedEC_ESM':
                x_train_model = np.concatenate((zero_shot_data['ec']['train'].T,zero_shot_data['esm']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['ec']['test'].T,zero_shot_data['esm']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['ec']['all'].T,zero_shot_data['esm']['all'].T,x_all), axis=1)
            elif model == 'AugmentedESM_Energy':
                x_train_model = np.concatenate((zero_shot_data['esm']['train'].T,zero_shot_data['energy']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['esm']['test'].T,zero_shot_data['energy']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['esm']['all'].T, zero_shot_data['energy']['all'].T,x_all), axis=1)
            elif model == 'AugmentedEC_Energy_ESM':
                x_train_model = np.concatenate((zero_shot_data['ec']['train'].T,zero_shot_data['energy']['train'].T,zero_shot_data['esm']['train'].T,x_train), axis=1)
                x_test_model = np.concatenate((zero_shot_data['ec']['test'].T,zero_shot_data['energy']['test'].T,zero_shot_data['esm']['test'].T,x_test), axis=1)
                x_all_model = np.concatenate((zero_shot_data['ec']['all'].T,zero_shot_data['energy']['all'].T,zero_shot_data['esm']['all'].T,x_all), axis=1)
            

            alphas_to_test = np.linspace(0.01, 100, 100)
            clf_cv = linear_model.RidgeCV(alphas=alphas_to_test, cv=None)
            hyper_tune = clf_cv.fit(x_train_model, y_train)
            tuned_alpha = hyper_tune.alpha_
            
            y_predict_test = hyper_tune.predict(x_test_model)
            y_predict_all = hyper_tune.predict(x_all_model)
            name = f"{model}: {encoding}"
            
            spearman_r = stats.spearmanr(y_actual, y_predict_test)[0]
            
            compare = pd.DataFrame({'actual':y_actual,'predicted':y_predict_test})
            predicted_sort = compare.sort_values('predicted', ascending=False)
            actual_sort = compare.sort_values('actual', ascending=False)
            DCG = 0
            for i,n in enumerate(predicted_sort['actual']):
                add = n/(np.log2(i+2))
                DCG += add
            ideal_DCG = 0
            for i,n in enumerate(actual_sort['actual']):
                add = n/(np.log2(i+2))
                ideal_DCG += add
            ndcg_calc = DCG/ideal_DCG
            
            return (name, spearman_r, ndcg_calc, tuned_alpha, y_predict_all, y_predict_test)

        jobs = list(product(model_list, encoding_list)) 
        
        try:
            # Try to use joblib, but check for SLURM_CPUS_PER_TASK
            n_cores_str = os.environ.get('SLURM_CPUS_PER_TASK')
            if n_cores_str is not None:
                n_cores = int(n_cores_str)
            else:
                n_cores = max(1, os.cpu_count() - 1)
        except Exception:
            n_cores = 1 # Default to 1 if something goes wrong

        results = Parallel(n_jobs=n_cores)(delayed(_fit_one_model)(job) for job in jobs)
        
        architecture = []
        spearman = []
        ndcg = []
        alpha = []
        
        combo = self._model_scope_mutations
        predictions = {'Mutation':combo}
        predictions_test = {'Mutation': self._test_data['AminoAcid'], 'Actual': self._test_data['Activity']}

        for result in results:
            if result is None: 
                continue
                
            (name, spearman_r, ndcg_calc, tuned_alpha, y_predict_all, y_predict_test) = result
            
            architecture.append(name)
            spearman.append(spearman_r)
            ndcg.append(ndcg_calc)
            alpha.append(tuned_alpha)
            predictions[name] = y_predict_all
            predictions_test[name] = y_predict_test
            
            if ShowPlots == True:
                plt.scatter(y_actual, y_predict_test)
                plt.title(name)
                plt.show()

        model_metrics = pd.DataFrame(data={'architecture':architecture, 'spearman_r':spearman, 'NDCG':ndcg, 'Tuned Alpha':alpha})
        predictions_df = pd.DataFrame(data=predictions)
        predictions_test_df = pd.DataFrame(data=predictions_test)
        
        self._all_predictions = predictions_df
        self._test_predictions = predictions_test_df
        self._model_metrics = model_metrics
        
        print(f"Parallel compare_and_predict finished in {time.time() - t0:.2f} seconds.")
        return
   
    
    def train_and_predict(self, model, encoding):
        """
        Trains one model.
        """
        alpha = []
        
        has_ec = hasattr(self, '_EC_predictions')
        has_energy = hasattr(self, '_Energy_predictions')
        has_esm = hasattr(self, '_ESM_predictions')

        if 'EC' in model and not has_ec:
            print(f"Error: Model '{model}' requires EC file, but it was not loaded.")
            return 
        if 'Energy' in model and not has_energy:
            print(f"Error: Model '{model}' requires Energy (Maestro) file, but it was not loaded.")
            return 
        if 'ESM' in model and not has_esm:
            print(f"Error: Model '{model}' requires ESM file, but it was not loaded.")
            return 
        
        combo = self._model_scope_mutations
        
        x_train = encode(self._training_data, self._wt_sequence, self._model_scope_mutations)[encoding]
        y_train = normalize_data(self._training_data, self._data_normalization_type)
        
        x_all = encode(pd.DataFrame(), self._wt_sequence, self._model_scope_mutations)[encoding]
        
        if "EC" in model:  
            ec_predictions_train = np.array([[self._EC_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            ec_predictions_all = np.array([[self._EC_predictions[mut] for mut in self._model_scope_mutations]]) # Aligned
        if "Energy" in model:
            energy_predictions_train = np.array([[self._Energy_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            energy_predictions_all = np.array([[self._Energy_predictions[mut] for mut in self._model_scope_mutations]]) # Aligned
        if "ESM" in model:
            esm_predictions_train = np.array([[self._ESM_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            esm_predictions_all = np.array([[self._ESM_predictions[mut] for mut in self._model_scope_mutations]]) # Aligned
        
        # Build the augmented model inputs
        x_train_model, x_all_model = x_train, x_all
        
        if model == 'Simple':
            pass # Already set
        elif model == 'AugmentedEC':
            x_train_model = np.concatenate((ec_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedEnergy':
            x_train_model = np.concatenate((energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((energy_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedESM':
            x_train_model = np.concatenate((esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((esm_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedEC_Energy':
            x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedEC_ESM':
            x_train_model = np.concatenate((ec_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedESM_Energy':
            x_train_model = np.concatenate((esm_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((esm_predictions_all.T, energy_predictions_all.T,x_all), axis=1)
        elif model == 'AugmentedEC_Energy_ESM':
            x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
        
        # Use RidgeCV for speed
        alphas_to_test = np.linspace(0.01, 100, 100)
        clf_cv = linear_model.RidgeCV(alphas=alphas_to_test, cv=None) 
        hyper_tune = clf_cv.fit(x_train_model, y_train)
        
        tuned_alpha = hyper_tune.alpha_
        self._best_alpha = tuned_alpha # Store the best alpha
        
        y_predict_all = hyper_tune.predict(x_all_model)
        predictions_df = pd.DataFrame(data={'Mutation':combo, 'Prediction':y_predict_all})
        
        self._predictions_df = predictions_df
        
        return
