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
import os, re

from Bio.PDB import MMCIFParser, ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial.distance import pdist, squareform
import warnings

def linregress(df, xcol, ycol, spearman=True):
    df = df[[xcol,ycol]].dropna()
    x = df[xcol]
    y = df[ycol]

    # Calculate the slope (m) and y-intercept (b) using linear regression
    # linregress returns slope, intercept, r_value, p_value, stderr
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate R-squared from the r_value
    r_squared = r_value**2
    
    if spearman:
        spear, p = stats.spearmanr(y,x)
        return slope, intercept, r_squared, spear, p
    else:
        return slope, intercept, r_squared

def analyze_protein_structure(cif_path):
    """
    Parses a .cif file, calculates SASA per residue, and generates 
    a pairwise Euclidean distance matrix based on Alpha Carbon atoms.

    Args:
        cif_path (str): Path to the .cif file.

    Returns:
        tuple: (sasa_df, distance_df)
            - sasa_df: DataFrame with residue IDs and SASA values.
            - distance_df: DataFrame containing the NxN distance matrix.
    """
    
    # 1. Parse the structure
    # QUIET=True suppresses warnings about minor PDB construction errors
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", cif_path)
    except Exception as e:
        return f"Error parsing file: {e}", None

    # We typically analyze the first model in the structure
    model = structure[0]

    # 2. Calculate SASA (Solvent Accessible Surface Area)
    # Using Shrake-Rupley algorithm (pure Python, no external binary needed)
    # probe_radius=1.40 is the standard for water
    sr = ShrakeRupley(probe_radius=1.40)
    sr.compute(model, level="R")  # level="R" computes SASA per Residue

    # 3. Extract Data
    residue_list = []
    sasa_values = []
    coords = []
    labels = []

    for chain in model:
        for residue in chain:
            # Filter: We generally only want standard Amino Acids, not water/ligands
            if is_aa(residue, standard=True):
                
                # Get Residue ID (Chain + Residue Number)
                res_id = residue.get_id()
                # res_id[1] is the residue number. res_id[2] is insertion code.
                label = f"{chain.id}:{residue.resname}{res_id[1]}"
                
                # Check for Alpha Carbon for distance calculation
                if 'CA' in residue:
                    # Get SASA (calculated by ShrakeRupley and stored in .sasa field)
                    sasa_val = residue.sasa
                    
                    # Get Coordinates of Alpha Carbon
                    coord = residue['CA'].get_coord()

                    residue_list.append(residue)
                    labels.append(label)
                    sasa_values.append(sasa_val)
                    coords.append(coord)
                else:
                    # Handle cases where CA is missing (rare in good structures)
                    warnings.warn(f"Skipping residue {label} due to missing Alpha Carbon.")

    # 4. Create SASA DataFrame
    sasa_df = pd.DataFrame({
        'Residue_ID': labels,
        'SASA': sasa_values
    })
    sasa_df.set_index('Residue_ID', inplace=True)

    # 5. Create Distance Matrix DataFrame
    if len(coords) > 0:
        # pdist calculates pairwise distances between all coordinates
        dist_array = pdist(coords, metric='euclidean')
        # squareform converts the compressed distance array to a full NxN matrix
        dist_matrix = squareform(dist_array)
        
        distance_df = pd.DataFrame(
            dist_matrix, 
            index=labels, 
            columns=labels
        )
    else:
        distance_df = pd.DataFrame()

    return sasa_df, distance_df

WT_SEQUENCE = "MGEKIEHPQWSYSGKTGPKYWGYLSGKTGPKYWGYLSPEYIMCAIGKNQSPIDLNEKYMVKACTRPLQINYVADAVKVLNNGHTIKVITLGKSYVVIDGRKFYLRQFHFHAPSEHTVNGEYYPFEAHFVHTDDEGNIAVIGVLFKLGKTNKELQKIWDYMPTKVGQENLLLTKVNPYLLLPKKKDYYRYNGSLTTPPCSEGVRWIIFKEPVEISAEQLNLFKEVMGFPNNRPIQPINARKILK"

#PATH TO DATA DIRECTORY
source = 'D:/Downloads/Files to work with/results'#r'C:\Users\hyerc\Downloads\Files to work with\results'#
perfect_tiling_path = f'{source}/perfect_tiling_path.tsv'
cif_path = 'D:/Downloads/Files to work with/results/CA_AF.cif'#r'C:\Users\hyerc\Downloads\Files to work with\results\CA_AF.cif'#

mutation_regex = re.compile(r"([A-Z])(\d+)([A-Z])")
struct_regex = re.compile(r"([A-Z]):([A-Z])([A-Z])([A-Z])(\d+)")

PHYSICAL_DESCRIPTORS = {
'A': [-3.11,    -2.90,	-1.03, 0],
'R': [3.66,	    2.41,	1.31, 1],
'N': [1.90,	    -0.68,	0.79, 0],
'D': [3.01,	    -0.92,	1.23, -1],
'C': [-0.08,	-1.89,	0.15, 0],
'Q': [2.85,	    0.36,	1.09, 0],
'E': [3.26,	    0.16,	1.28, -1],
'G': [-0.30,	-4.04,	0.01, 0],
'H': [3.03,	    0.83,	1.15, 1],
'I': [-3.53,	0.51,	-1.32, 0],
'L': [-3.77,	0.52,	-1.40, 0],
'K': [3.50,	    0.92,	1.23, 1],
'M': [-4.06,	0.92,	-1.42, 0],
'F': [-4.06,	2.22,	-1.47, 0],
'P': [-1.93,	-1.25,	-0.64, 0],
'S': [0.70,	    -2.36,	0.38, 0],
'T': [0.56,	    -1.19,	0.28, 0],
'W': [-0.50,	4.28,	-0.18, 0],
'Y': [-0.59,	2.75,	-0.18, 0],
'V': [-3.53,	-0.65,	-1.27, 0]
}

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

sasa_df, distance_df = analyze_protein_structure(cif_path)

old_dist_cols = list(distance_df.columns)
new_dist_cols = [int(struct_regex.match(col).groups()[-1]) for col in old_dist_cols]

old_dist_index = list(distance_df.index)
new_dist_index = [int(struct_regex.match(ind).groups()[-1]) for ind in old_dist_index]

distance_df.rename(columns=dict(zip(old_dist_cols,new_dist_cols)),inplace=True)
distance_df.rename(index=dict(zip(old_dist_index,new_dist_index)),inplace=True)

old_sasa_index = list(sasa_df.index)
new_sasa_index = [int(struct_regex.match(ind).groups()[-1]) for ind in old_sasa_index]
sasa_df.rename(index=dict(zip(old_sasa_index,new_sasa_index)),inplace=True)

def choice_attributes(predictions, ptDF, WT_SEQUENCE, sasa_df, distance_df):
    act_cols = ['Iteration','Source Activity','LB','Q1','Mean','Median','Q3','HB']
    activityDF = pd.DataFrame(columns=act_cols)
    err_cols = ['Iteration','Source Error', 'Train Error','LB','Q1','Mean','Median','Q3','HB','Std']
    errorDF = pd.DataFrame(columns=err_cols)
    physical_cols = ['Iteration','Mutation','Residue Number','dPolarity','dVolume','dHydrophilicity','dCharge']
    physicalDF = pd.DataFrame(columns=physical_cols)
    aa_columns = [int(i + 1) for i in range(len(WT_SEQUENCE))]
    location_cols = ['Iteration','Residue Number'] + aa_columns + [f'{col}_d' for col in aa_columns] + [f'{col}_n' for col in aa_columns]
    locationDF = pd.DataFrame(columns=location_cols)
    
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
        std = test['Error'].std()
        errorDF = errorDF._append(pd.Series(dict(zip(err_cols,[iteration,source_error,train_error,lb,q1,mean,median,q3,hb,std]))),ignore_index=True)
        
        #Amino Acid Comparison: What kind of mutation is happening?
        aa1, rn, aa2 = mutation_regex.match(mutation).groups()
        rn = int(rn)
        sasa = sasa_df.loc[rn]['SASA']
        orig_polarity, orig_volume, orig_hydrophilicity, orig_charge = PHYSICAL_DESCRIPTORS[aa1]
        new_polarity, new_volume, new_hydrophilicity, new_charge = PHYSICAL_DESCRIPTORS[aa1]
        dpolarity = new_polarity - orig_polarity
        dvolume = new_volume - orig_volume
        dhydrophilicity = new_hydrophilicity - orig_hydrophilicity
        dcharge = new_charge - orig_charge
        physicalDF = physicalDF._append(pd.Series(dict(zip(physical_cols,[iteration,mutation,rn,dpolarity,dvolume,dhydrophilicity,dcharge,sasa]))),ignore_index=True)
        
        #Location Comparison: Where is the model picking from? How does this impact error at nearby vs far residues? (Try euclidean distance method)
        new = new[new['Class']=='test']
        new.reset_index(inplace=True)
        new['Residue Number'] = new['AminoAcid'].str[1:-1].astype(int)
        prediction['Residue Number'] = prediction['AminoAcid'].str[1:-1].astype(int)
        old_error = prediction[['Residue Number','Error']].groupby(by='Residue Number').median().reset_index().rename(columns={'Error':'Old_Error'})
        new_error = new[['Residue Number','Error']].groupby(by='Residue Number').median().reset_index().rename(columns={'Error':'New_Error'})
        conDF = new_error.merge(old_error,how='left',on='Residue Number')
        conDF['dError'] = conDF['New_Error'] - conDF['Old_Error']
        min_val = conDF['dError'].min() * -1
        max_val = conDF['dError'].max() * -1

        # Apply min-max normalization
        conDF['Normalized'] = (conDF['dError'] * -1 - min_val) / (max_val - min_val)
        error = dict(zip(conDF['Residue Number'],conDF['dError']))
        normalized_error = dict(zip(conDF['Residue Number'].astype(str) + '_n',conDF['Normalized']))
        distances = distance_df[rn]
        distances = dict(zip([f'{ind}_d' for ind in distances.index],list(distances)))
        to_append = {'Iteration':iteration,'Residue Number':rn}
        to_append.update(error)
        to_append.update(normalized_error)
        to_append.update(distances)
        missing = [col for col in locationDF.columns if col not in list(to_append.keys())]
        to_append.update(dict(zip(missing,np.full(len(missing),np.nan))))
        locationDF = locationDF._append(pd.Series(to_append),ignore_index=True)        
                
    return activityDF, errorDF, physicalDF, locationDF

activityDF, errorDF, physicalDF, locationDF = choice_attributes(predictions, ptDF, WT_SEQUENCE, sasa_df, distance_df)

def correlation_over_iterations(predictions):
    corr_cols = ['Iteration','m_test','b_test','rsq_test','Spearman_test','p_test','m_train','b_train','rsq_train','Spearman_train','p_train','Median Error Test','Median Error Train','Median Error Test Transform','Median Error Train Transform']
    corrDF = pd.DataFrame(columns=corr_cols)
    for iteration, prediction in predictions.items():
        test = prediction[prediction['Class'] == 'test'].dropna()
        train = prediction[prediction['Class'] == 'train'].dropna()
        test_m, test_b, test_rsq, test_spear, test_p = linregress(test, 'Prediction', 'Activity')
        train_m, train_b, train_rsq, train_spear, train_p = linregress(train, 'Prediction', 'Activity')
        test_med = test['Error'].median()
        train_med = train['Error'].median()
        transform = (test['Prediction'] * train_m) - train_b
        transform_error = np.abs(transform - test['Activity'])
        transform_med = transform_error.median()
        transform_train = (train['Prediction'] * train_m) - train_b
        transform_error_train = np.abs(transform_train - train['Activity'])
        transform_med_train = transform_error_train.median()
        to_append = dict(zip(corr_cols,[iteration, test_m, test_b, test_rsq, test_spear, test_p, train_m, train_b, train_rsq, train_spear, train_p, test_med, train_med, transform_med, transform_med_train]))
        corrDF = corrDF._append(pd.Series(to_append),ignore_index=True)
    return corrDF

corrDF = correlation_over_iterations(predictions)
corrDF = corrDF.sort_values(by='Iteration')