# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:54:09 2022

@author: gmlan
"""
#code inspired by the wonderful https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data 
#and https://github.com/fhalab/MLDE

from src.AugmentedMLDE_HelperFuncs import encode, normalize_data, ALL_AAS
import pandas as pd
import numpy as np
from itertools import product
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

model_list = ['Simple', 'AugmentedESM', 'AugmentedEC', 'AugmentedEnergy', 'AugmentedEC_Energy', 'AugmentedESM_Energy', 'AugmentedEC_ESM', 'AugmentedEC_Energy_ESM']
encoding_list = ['One Hot', 'Georgiev', 'ZScales', 'VHSE', 'Physical Descriptors'] 

class AugmentedMLDEmodel():
    """
    Class that handles trainining and predicting for augmented ridge regression models
    
    Inputs:
    training_data_file, test_data_file
        manually curated data sets for model training and validation; training_data_file contained single mutants, while test_data_file contains higher order mutants
    num_positions
        the number of hot spots selected from the hot spot screen to be combinatorially mutated
    EC_file, Maestro_file, ESM_file
        contain zero-shot predictions for the entire combinatorial space
    
    Functions:
    self.compare_and_predict()
        Used to identify the best augmented model by testing combinations of zero-shot predictors and encoding strategies (requires test and train data files)
    self.train_and_predict()
        Given a specific augmented model (identified in compare_and_predict), return predictions for the entire combinatorial space
    """
    
    def __init__(self, training_data_file, test_data_file, sequence_length, wt_sequence, EC_file, Maestro_file, ESM_file):
        self._training_data_file = training_data_file
        self._test_data_file = test_data_file
        self._EC_file = EC_file
        self._Maestro_file = Maestro_file
        self._ESM_file = ESM_file
        self._sequence_length = sequence_length
        self._wt_sequence = wt_sequence
        self._data_normalization_type = 'Standardization'
        self._random_seed = 42
        self._regularization_coeff = 10**-8
        self._all_predictions = None
        self._test_predictions = None
        self._model_metrics = None
        self._predictions_df = None
        
        training_data = pd.read_excel(self._training_data_file, keep_default_na=False)
        self._training_data = training_data
        
        if self._test_data_file != None:
            test_data = pd.read_excel(self._test_data_file, keep_default_na=False)
            self._test_data = test_data
    
        def prep_data(file_loc):
            """
            Standardizes unsupervised zero-shot predictions and scales according to a given regularization coefficient 
            """
            sc = StandardScaler()
            raw = pd.read_csv(file_loc)
            raw.sort_values('Mutations', inplace=True)
            raw.set_index('Mutations', inplace=True)
            raw_scaled = sc.fit_transform(np.array(raw['Predictions']).reshape(-1,1))
            raw['Scaled Predictions'] = raw_scaled
            raw['Regularized Features'] = -1 * raw['Scaled Predictions'] * np.sqrt(1 / self._regularization_coeff)
            return raw['Regularized Features']
        
        if self._EC_file != None:
            self._EC_predictions = prep_data(self._EC_file)
        if self._Maestro_file != None:
            self._Energy_predictions = prep_data(self._Maestro_file)
        if self._ESM_file != None:
            self._ESM_predictions = prep_data(self._ESM_file)
          
   
    def compare_and_predict(self, ShowPlots = False):
        """
        Given a train and test data set, calculates NDCG and spearman correlation for a variety of zeroshot predictors and encodings.
        
        Returns
        -------
            self._all_predictions - model predications for the entire combinatorial space
            self._test_predictions - model predictions for the test data set
            self._model_metrics - spearman_r and ndcg for predictions made on the withheld test data set
        """
        architecture = []
        spearman = []
        ndcg = []
        alpha = []
        
        all_combos_list = []
        for i in range(self._sequence_length):
            pos_1_indexed = i + 1
            original_aa = self._wt_sequence[i]
            for new_aa in ALL_AAS:
                mutation_string = f"{original_aa}{pos_1_indexed}{new_aa}"
                all_combos_list.append(mutation_string)
        combo = all_combos_list

        predictions = {'Mutation':combo}
        predictions_test = {'Mutation': self._test_data['AminoAcid'], 'Actual': self._test_data['Activity']}
        
        for encoding in encoding_list:       
            #encode and normalize training data
            x_train = encode(self._training_data, self._sequence_length, self._wt_sequence)[encoding]
            y_train = normalize_data(self._training_data, self._data_normalization_type)
            
            #data and encodings to predict withheld ISM data (test data)
            x_test = encode(self._test_data, self._sequence_length, self._wt_sequence)[encoding]
            y_actual = normalize_data(self._test_data, self._data_normalization_type)
            
            #encodings to predict entire combinatorial space
            x_all = encode(pd.DataFrame(), self._sequence_length, self._wt_sequence)[encoding]
            
            #generating encodings for augmented models using zero shot predictions
            ec_predictions_train = np.array([[self._EC_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            ec_predictions_test = np.array([[self._EC_predictions.loc[aa] for aa in self._test_data['AminoAcid']]])
            ec_predictions_all = np.array([self._EC_predictions])

            energy_predictions_train = np.array([[self._Energy_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            energy_predictions_test = np.array([[self._Energy_predictions.loc[aa] for aa in self._test_data['AminoAcid']]])    
            energy_predictions_all = np.array([self._Energy_predictions])

            esm_predictions_train = np.array([[self._ESM_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            esm_predictions_test = np.array([[self._ESM_predictions.loc[aa] for aa in self._test_data['AminoAcid']]])
            esm_predictions_all = np.array([self._ESM_predictions])
            
            for model in model_list:
                if model == 'Simple':
                    x_train_model = x_train
                    x_test_model = x_test
                    x_all_model = x_all
                if model == 'AugmentedEC':
                    x_train_model = np.concatenate((ec_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((ec_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((ec_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedEnergy':
                    x_train_model = np.concatenate((energy_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((energy_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((energy_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedESM':
                    x_train_model = np.concatenate((esm_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((esm_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((esm_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedEC_Energy':
                    x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((ec_predictions_test.T,energy_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedEC_ESM':
                    x_train_model = np.concatenate((ec_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((ec_predictions_test.T,esm_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((ec_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedESM_Energy':
                    x_train_model = np.concatenate((esm_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((esm_predictions_test.T,energy_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((esm_predictions_all.T, energy_predictions_all.T,x_all), axis=1)
                if model == 'AugmentedEC_Energy_ESM':
                    x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
                    x_test_model = np.concatenate((ec_predictions_test.T,energy_predictions_test.T,esm_predictions_test.T,x_test), axis=1)
                    x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
                
                #hyperparameter tuning of ridge regression model using k-fold cv of training data
                cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=self._random_seed)
                clf = linear_model.Ridge()
                parameters = {'alpha':np.linspace(0.01, 100, 100)}
                search = GridSearchCV(clf, parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, verbose=False)
                hyper_tune = search.fit(x_train_model, y_train)
                tuned_alpha = hyper_tune.best_estimator_
                alpha.append(tuned_alpha)
                
                #make focused predictions on withheld test data and plot actual vs. predictions
                y_predict_test = hyper_tune.predict(x_test_model)
                name = f"{model}: {encoding}" 
                predictions_test[name] = y_predict_test
                if ShowPlots == True:
                    plt.scatter(y_actual, y_predict_test)
                    plt.title(name)
                    plt.show()
                
                #make predictions of entire combinatorial data set
                y_predict_all = hyper_tune.predict(x_all_model)
                predictions[name] = y_predict_all
                architecture.append(name)
                
                #calculate spearman correlation coefficiant and NDCG
                spearman_r = stats.spearmanr(y_actual, y_predict_test)[0]
                spearman.append(spearman_r)
                
                #for NDCG, first rank order actual data set and align this with predicted values
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
                ndcg.append(ndcg_calc)
                
        model_metrics = pd.DataFrame(data={'architecture':architecture, 'spearman_r':spearman, 'NDCG':ndcg, 'Tuned Alpha':alpha})
        
        predictions_df = pd.DataFrame(data=predictions)
        predictions_test_df = pd.DataFrame(data=predictions_test)
        
        self._all_predictions = predictions_df
        self._test_predictions = predictions_test_df
        self._model_metrics = model_metrics
            
        return
   
    
    def train_and_predict(self, model, encoding):
        """
        Given a specific augmentation strategy and encoding, train a model on a training data set and make predictions of the entire combinatorial space. 
        
        Returns
        -------
            self._predictions_df - model predictions for the entire combinatorial space
        """
        alpha = []
        
        all_combos_list = []
        for i in range(self._sequence_length):
            pos_1_indexed = i + 1
            original_aa = self._wt_sequence[i]
            for new_aa in ALL_AAS:
                mutation_string = f"{original_aa}{pos_1_indexed}{new_aa}"
                all_combos_list.append(mutation_string)
        
        combo = all_combos_list
        
        #generate all encodings for the specified model and encoding type
        #encode and normalize training data
        x_train = encode(self._training_data, self._num_positions)[encoding]
        y_train = normalize_data(self._training_data, self._data_normalization_type)
        
        #encodings to predict entire combinatorial space
        x_all = encode(pd.DataFrame(), self._num_positions)[encoding]
        
        #generating encodings for augmented models using zero shot predictions
        if "EC" in model:  
            ec_predictions_train = np.array([[self._EC_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            ec_predictions_all = np.array([self._EC_predictions])

        if "Energy" in model:
            energy_predictions_train = np.array([[self._Energy_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            energy_predictions_all = np.array([self._Energy_predictions])

        if "ESM" in model:
            esm_predictions_train = np.array([[self._ESM_predictions.loc[aa] for aa in self._training_data['AminoAcid']]])
            esm_predictions_all = np.array([self._ESM_predictions])
        
        if model == 'Simple':
            x_train_model = x_train
            x_all_model = x_all
        if model == 'AugmentedEC':
            x_train_model = np.concatenate((ec_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedEnergy':
            x_train_model = np.concatenate((energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((energy_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedESM':
            x_train_model = np.concatenate((esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((esm_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedEC_Energy':
            x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedEC_ESM':
            x_train_model = np.concatenate((ec_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedESM_Energy':
            x_train_model = np.concatenate((esm_predictions_train.T,energy_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((esm_predictions_all.T, energy_predictions_all.T,x_all), axis=1)
        if model == 'AugmentedEC_Energy_ESM':
            x_train_model = np.concatenate((ec_predictions_train.T,energy_predictions_train.T,esm_predictions_train.T,x_train), axis=1)
            x_all_model = np.concatenate((ec_predictions_all.T,energy_predictions_all.T,esm_predictions_all.T,x_all), axis=1)
        
        #hyperparameter tuning of ridge regression model using k-fold cv of all training data
        cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=self._random_seed)
        clf = linear_model.Ridge()
        parameters = {'alpha':np.linspace(0.01, 100, 100)}
        search = GridSearchCV(clf, parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, verbose=True)
        hyper_tune = search.fit(x_train_model, y_train)
        tuned_alpha = hyper_tune.best_estimator_
        alpha.append(tuned_alpha)
        MSE = hyper_tune.best_score_
        
        #make predictions of entire combinatorial data set
        y_predict_all = hyper_tune.predict(x_all_model)
        predictions_df = pd.DataFrame(data={'Mutation':combo, 'Prediction':y_predict_all})
        
        self._predictions_df = predictions_df
        
        return
        
