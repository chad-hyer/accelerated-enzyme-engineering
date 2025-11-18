# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:53:38 2025

@author: hyerc

Scripts for taking data from tiling experiments to elucidate reasons behind tiling decisions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"

#PATH TO DATA DIRECTORY
source = 'C:/Users/hyerc/Downloads/Files to work with/results'
perfect_tiling_path = f'{source}/perfect_tiling_path.tsv'

#DATA FILES EXTRACTION
folders = [folder for folder in os.listdir(source) if 'it' in folder]
iterations = [int(folder.replace('it_','')) for folder in folders]
error_matrices_files = [pd.read_csv(f'{source}/{folder}/best_error_{folder}.csv') for folder in folders]
median_error_files = [pd.read_csv(f'{source}/{folder}/median_error_{folder}.csv') for folder in folders]
prediction_files = [pd.read_csv(f'{source}/{folder}/prediction_{folder}.csv') for folder in folders]

error_matrices = dict(zip(iterations, error_matrices_files))
median_errors = dict(zip(iterations, median_error_files))
predictions = dict(zip(iterations, prediction_files))
ptDF = pd.read_csv(perfect_tiling_path,delimiter='\t')

def choice_attributes(predictions, ptDF):
    act_cols = ['Iteration','Source Activity','LB','Q1','Mean','Median','Q3','HB']
    activityDF = pd.DataFrame(columns=act_cols)
    err_cols = ['Iteration','Source Error', 'Train Error','LB','Q1','Mean','Median','Q3','HB']
    errorDF = pd.DataFrame(columns=err_cols)
    for index, row in ptDF.iterrows():
        iteration = row['Iteration']
        mutation = row['Best_Mutation']
        if iteration == 1: continue #skip first choice to avoid errors if it_0 does not exist
        prediction = predictions[iteration - 1].copy() #get original prediction that potentially informed the iteration choice
        new = predictions[iteration].copy()
        new.set_index('AminoAcid', inplace=True)
        new.dropna(inplace=True)
        test = prediction[prediction['Class'] == 'test']
        test.set_index('AminoAcid', inplace=True)
        try:
            source_mutation = test.loc[mutation]
            train_mutation = new.loc[mutation]
        except KeyError:
            print(f'Source Mutation ({mutation}) does not exist.')
            continue
        
        #Activity Comparison: What is the real activity of the data added?
        source_activity = source_mutation['Activity']
        q1 = test['Activity'].quantile(0.25)
        q3 = test['Activity'].quantile(0.75)
        mean = test['Activity'].mean()
        median = test['Activity'].median()
        iqr = q3 - q1
        lb = q1 - (1.5 * iqr)
        hb = q3 + (1.5 * iqr)
        activityDF = activityDF._append(pd.Series(dict(zip(act_cols,[iteration,source_activity,lb,q1,mean,median,q3,hb]))),ignore_index=True)
        
        
        #Error Comparison: What is the error of the previous prediction on the data added?
        source_error = source_mutation['Error']
        train_error = train_mutation['Error']
        q1 = test['Error'].quantile(0.25)
        q3 = test['Error'].quantile(0.75)
        mean = test['Error'].mean()
        median = test['Error'].median()
        iqr = q3 - q1
        lb = q1 - (1.5 * iqr)
        hb = q3 + (1.5 * iqr)
        errorDF = errorDF._append(pd.Series(dict(zip(err_cols,[iteration,source_error,train_error,lb,q1,mean,median,q3,hb]))),ignore_index=True)
        
        #Amino Acid Comparison: What kind of mutation is happening?
        
        
        #Location Comparison: Where is the model picking from? How does this impact error at nearby vs far residues? (Try euclidean distance method)
    return activityDF, errorDF