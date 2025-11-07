# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 17:23:19 2022

@author: gmlan
"""

import numpy as np
from itertools import product
import re

ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

# From ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345)
gg_1 = {'Q': -2.54, 'L': 2.72, 'T': -0.65, 'C': 2.66, 'I': 3.1, 'G': 0.15, 'V': 2.64, 'K': -3.89, 'M': 1.89, 'F': 3.12, 'N': -2.02, 'R': -2.8, 'H': -0.39, 'E': -3.08, 'W': 1.89, 'A': 0.57, 'D': -2.46, 'Y': 0.79, 'S': -1.1, 'P': -0.58}
gg_2 = {'Q': 1.82, 'L': 1.88, 'T': -1.6, 'C': -1.52, 'I': 0.37, 'G': -3.49, 'V': 0.03, 'K': 1.47, 'M': 3.88, 'F': 0.68, 'N': -1.92, 'R': 0.31, 'H': 1, 'E': 3.45, 'W': -0.09, 'A': 3.37, 'D': -0.66, 'Y': -2.62, 'S': -2.05, 'P': -4.33}
gg_3 = {'Q': -0.82, 'L': 1.92, 'T': -1.39, 'C': -3.29, 'I': 0.26, 'G': -2.97, 'V': -0.67, 'K': 1.95, 'M': -1.57, 'F': 2.4, 'N': 0.04, 'R': 2.84, 'H': -0.63, 'E': 0.05, 'W': 4.21, 'A': -3.66, 'D': -0.57, 'Y': 4.11, 'S': -2.19, 'P': -0.02}
gg_4 = {'Q': -1.85, 'L': 5.33, 'T': 0.63, 'C': -3.77, 'I': 1.04, 'G': 2.06, 'V': 2.34, 'K': 1.17, 'M': -3.58, 'F': -0.35, 'N': -0.65, 'R': 0.25, 'H': -3.49, 'E': 0.62, 'W': -2.77, 'A': 2.34, 'D': 0.14, 'Y': -0.63, 'S': 1.36, 'P': -0.21}
gg_5 = {'Q': 0.09, 'L': 0.08, 'T': 1.35, 'C': 2.96, 'I': -0.05, 'G': 0.7, 'V': 0.64, 'K': 0.53, 'M': -2.55, 'F': -0.88, 'N': 1.61, 'R': 0.2, 'H': 0.05, 'E': -0.49, 'W': 0.72, 'A': -1.07, 'D': 0.75, 'Y': 1.89, 'S': 1.78, 'P': -8.31}
gg_6 = {'Q': 0.6, 'L': 0.09, 'T': -2.45, 'C': -2.23, 'I': -1.18, 'G': 7.47, 'V': -2.01, 'K': 0.1, 'M': 2.07, 'F': 1.62, 'N': 2.08, 'R': -0.37, 'H': 0.41, 'E': 0, 'W': 0.86, 'A': -0.4, 'D': 0.24, 'Y': -0.53, 'S': -3.36, 'P': -1.82}
gg_7 = {'Q': 0.25, 'L': 0.27, 'T': -0.65, 'C': 0.44, 'I': -0.21, 'G': 0.41, 'V': -0.33, 'K': 4.01, 'M': 0.84, 'F': -0.15, 'N': 0.4, 'R': 3.81, 'H': 1.61, 'E': -5.66, 'W': -1.07, 'A': 1.23, 'D': -5.15, 'Y': -1.3, 'S': 1.39, 'P': -0.12}
gg_8 = {'Q': 2.11, 'L': -4.06, 'T': 3.43, 'C': -3.49, 'I': 3.45, 'G': 1.62, 'V': 3.93, 'K': -0.01, 'M': 1.85, 'F': -0.41, 'N': -2.47, 'R': 0.98, 'H': -0.6, 'E': -0.11, 'W': -1.66, 'A': -2.32, 'D': -1.17, 'Y': 1.31, 'S': -1.21, 'P': -1.18}
gg_9 = {'Q': -1.92, 'L': 0.43, 'T': 0.34, 'C': 2.22, 'I': 0.86, 'G': -0.47, 'V': -0.21, 'K': -0.26, 'M': -2.05, 'F': 4.2, 'N': -0.07, 'R': 2.43, 'H': 3.55, 'E': 1.49, 'W': -5.87, 'A': -2.01, 'D': 0.73, 'Y': -0.56, 'S': -2.83, 'P': 0}
gg_10 = {'Q': -1.67, 'L': -1.2, 'T': 0.24, 'C': -3.78, 'I': 1.98, 'G': -2.9, 'V': 1.27, 'K': -1.66, 'M': 0.78, 'F': 0.73, 'N': 7.02, 'R': -0.99, 'H': 1.52, 'E': -2.26, 'W': -0.66, 'A': 1.31, 'D': 1.5, 'Y': -0.95, 'S': 0.39, 'P': -0.66}
gg_11 = {'Q': 0.7, 'L': 0.67, 'T': -0.53, 'C': 1.98, 'I': 0.89, 'G': -0.98, 'V': 0.43, 'K': 5.86, 'M': 1.53, 'F': -0.56, 'N': 1.32, 'R': -4.9, 'H': -2.28, 'E': -1.62, 'W': -2.49, 'A': -1.14, 'D': 1.51, 'Y': 1.91, 'S': -2.92, 'P': 0.64}
gg_12 = {'Q': -0.27, 'L': -0.29, 'T': 1.91, 'C': -0.43, 'I': -1.67, 'G': -0.62, 'V': -1.71, 'K': -0.06, 'M': 2.44, 'F': 3.54, 'N': -2.44, 'R': 2.09, 'H': -3.12, 'E': -3.97, 'W': -0.3, 'A': 0.19, 'D': 5.61, 'Y': -1.26, 'S': 1.27, 'P': -0.92}
gg_13 = {'Q': -0.99, 'L': -2.47, 'T': 2.66, 'C': -1.03, 'I': -1.02, 'G': -0.11, 'V': -2.93, 'K': 1.38, 'M': -0.26, 'F': 5.25, 'N': 0.37, 'R': -3.08, 'H': -1.45, 'E': 2.3, 'W': -0.5, 'A': 1.66, 'D': -3.85, 'Y': 1.57, 'S': 2.86, 'P': -0.37}
gg_14 = {'Q': -1.56, 'L': -4.79, 'T': -3.07, 'C': 0.93, 'I': -1.21, 'G': 0.15, 'V': 4.22, 'K': 1.78, 'M': -3.09, 'F': 1.73, 'N': -0.89, 'R': 0.82, 'H': -0.77, 'E': -0.06, 'W': 1.64, 'A': 4.39, 'D': 1.28, 'Y': 0.2, 'S': -1.88, 'P': 0.17}
gg_15 = {'Q': 6.22, 'L': 0.8, 'T': 0.2, 'C': 1.43, 'I': -1.78, 'G': -0.53, 'V': 1.06, 'K': -2.71, 'M': -1.39, 'F': 2.14, 'N': 3.13, 'R': 1.32, 'H': -4.18, 'E': -0.35, 'W': -0.72, 'A': 0.18, 'D': -1.98, 'Y': -0.76, 'S': -2.42, 'P': 0.36}
gg_16 = {'Q': -0.18, 'L': -1.43, 'T': -2.2, 'C': 1.45, 'I': 5.71, 'G': 0.35, 'V': -1.31, 'K': 1.62, 'M': -1.02, 'F': 1.1, 'N': 0.79, 'R': 0.69, 'H': -2.91, 'E': 1.51, 'W': 1.75, 'A': -2.6, 'D': 0.05, 'Y': -5.19, 'S': 1.75, 'P': 0.08}
gg_17 = {'Q': 2.72, 'L': 0.63, 'T': 3.73, 'C': -1.15, 'I': 1.54, 'G': 0.3, 'V': -1.97, 'K': 0.96, 'M': -4.32, 'F': 0.68, 'N': -1.54, 'R': -2.62, 'H': 3.37, 'E': -2.29, 'W': 2.73, 'A': 1.49, 'D': 0.9, 'Y': -2.56, 'S': -2.77, 'P': 0.16}
gg_18 = {'Q': 4.35, 'L': -0.24, 'T': -5.46, 'C': -1.64, 'I': 2.11, 'G': 0.32, 'V': -1.21, 'K': -1.09, 'M': -1.34, 'F': 1.46, 'N': -1.71, 'R': -1.49, 'H': 1.87, 'E': -1.47, 'W': -2.2, 'A': 0.46, 'D': 1.38, 'Y': 2.87, 'S': 3.36, 'P': -0.34}
gg_19 = {'Q': 0.92, 'L': 1.01, 'T': -0.73, 'C': -1.05, 'I': -4.18, 'G': 0.05, 'V': 4.77, 'K': 1.36, 'M': 0.09, 'F': 2.33, 'N': -0.25, 'R': -2.57, 'H': 2.17, 'E': 0.15, 'W': 0.9, 'A': -4.22, 'D': -0.03, 'Y': -3.43, 'S': 2.67, 'P': 0.04}
GEORGIEV_PARAMETERS = [gg_1, gg_2, gg_3, gg_4, gg_5, gg_6, gg_7, gg_8, gg_9, gg_10, gg_11, gg_12, gg_13, gg_14, gg_15, gg_16, gg_17, gg_18, gg_19]
    
## ZScales from https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
ZSCALE = {
'A': [0.24,  -2.32,  0.60, -0.14,  1.30], 
'C': [0.84,  -1.67,  3.71,  0.18, -2.65], 
'D': [3.98,   0.93,  1.93, -2.46,  0.75], 
'E': [3.11,   0.26, -0.11, -0.34, -0.25], 
'F': [-4.22,  1.94,  1.06,  0.54, -0.62], 
'G': [2.05,  -4.06,  0.36, -0.82, -0.38], 
'H': [2.47,   1.95,  0.26,  3.90,  0.09], 
'I': [-3.89, -1.73, -1.71, -0.84,  0.26], 
'K': [2.29,   0.89, -2.49,  1.49,  0.31], 
'L': [-4.28, -1.30, -1.49, -0.72,  0.84], 
'M': [-2.85, -0.22,  0.47,  1.94, -0.98], 
'N': [3.05,   1.62,  1.04, -1.15,  1.61], 
'P': [-1.66,  0.27,  1.84,  0.70,  2.00], 
'Q': [1.75,   0.50, -1.44, -1.34,  0.66], 
'R': [3.52,   2.50, -3.50,  1.99, -0.17], 
'S': [2.39,  -1.07,  1.15, -1.39,  0.67], 
'T': [0.75,  -2.18, -1.12, -1.46, -0.40], 
'V': [-2.59, -2.64, -1.54, -0.85, -0.02], 
'W': [-4.36,  3.94,  0.59,  3.44, -1.59], 
'Y': [-2.54,  2.44,  0.43,  0.04, -1.47],
}

## VHSE from https://onlinelibrary.wiley.com/doi/full/10.1002/bip.20296?casa_token=RJ_e2svNRBEAAAAA%3Abp-IKnSYn_SzaHb5GvlsQymx4gZNSGQ39rSb_ckM3x5d1EBB-gqW5vZr16SHWiHa_v6cS0_ofT81VVm3
VHSE = {
'A': [0.15,  -1.11,	-1.35,	-0.92,	0.02,	-0.91,	0.36,	-0.48],
'R': [-1.47, 1.45,	1.24,	1.27,	1.55,	1.47,	1.3,	0.83],
'N': [-0.99, 0,	    -0.37,	0.69,	-0.55,	0.85,	0.73,	-0.80],
'D': [-1.15, 0.67,	-0.41,	-0.01,	-2.68,	1.31,	0.03,	0.56],
'C': [0.18,  -1.67,	-0.46,	-0.21,	0,    	1.2,	-1.61,	-0.19],
'Q': [-0.96, 0.12,	0.18,	0.16,	0.09,	0.42,	-0.20,	-0.41],
'E': [-1.18, 0.4,	0.1,	0.36,	-2.16,	-0.17,	0.91,	0.02],
'G': [-0.20, -1.53,	-2.63,	2.28,	-0.53,	-1.18,	2.01,	-1.34],
'H': [-0.43, -0.25,	0.37,	0.19,	0.51,	1.28,	0.93,	0.65],
'I': [1.27,  -0.14,	0.3,	-1.80,	0.3,	-1.61,	-0.16,	-0.13],
'L': [1.36,	 0.07,	0.26,	-0.80,	0.22,	-1.37,	0.08,	-0.62],
'K': [-1.17, 0.7,	0.7,	0.8,	1.64,	0.67,	1.63,	0.13],
'M': [1.01,	 -0.53,	0.43,	0,	    0.23,	0.1,	-0.86,	-0.68],
'F': [1.52,	 0.61,	0.96,	-0.16,	0.25,	0.28,	-1.33,	-0.20],
'P': [0.22,	 -0.17,	-0.50,	0.05,	-0.01,	-1.34,	-0.19,	3.56],
'S': [-0.67, -0.86,	-1.07,	-0.41,	-0.32,	0.27,	-0.64,	0.11],
'T': [-0.34, -0.51,	-0.55,	-1.06,	-0.06,	-0.01,	-0.79,	0.39],
'W': [1.5,	 2.06,	1.79,	0.75,	0.75,	-0.13,	-1.01,	-0.85],
'Y': [0.61,	 1.6,	1.17,	0.73,	0.53,	0.25,	-0.96,	-0.52],
'V': [0.76,	 -0.92,	-0.17,	-1.91,	0.22,	-1.40,	-0.24,	-0.03]
}

#Pyhsical descriptors from https://pubs.acs.org/doi/full/10.1021/acs.jcim.7b00488
PYHSICAL_DESCRIPTORS = {
'A': [-3.11,    -2.90,	-1.03],
'R': [3.66,	    2.41,	1.31],
'N': [1.90,	    -0.68,	0.79],
'D': [3.01,	    -0.92,	1.23],
'C': [-0.08,	-1.89,	0.15],
'Q': [2.85,	    0.36,	1.09],
'E': [3.26,	    0.16,	1.28],
'G': [-0.30,	-4.04,	0.01],
'H': [3.03,	    0.83,	1.15],
'I': [-3.53,	0.51,	-1.32],
'L': [-3.77,	0.52,	-1.40],
'K': [3.50,	    0.92,	1.23],
'M': [-4.06,	0.92,	-1.42],
'F': [-4.06,	2.22,	-1.47],
'P': [-1.93,	-1.25,	-0.64],
'S': [0.70,	    -2.36,	0.38],
'T': [0.56,	    -1.19,	0.28],
'W': [-0.50,	4.28,	-0.18],
'Y': [-0.59,	2.75,	-0.18],
'V': [-3.53,	-0.65,	-1.27]
}



# Normalize_encodings() and encode() are adapted from excellent work done by Witman et. al 
#from https://www.cell.com/cell-systems/pdfExtended/S2405-4712(21)00286-6: 
def normalize_encodings(unnormalized_encodings):
    """
    Takes a tensor of embeddings, flattens the internal dimensions, then mean-centers
    and unit-scales. After scaling, the matrix is repacked to its original shape.
    
    Parameters
    ----------
    unnormalized_encodings: Numpy array of shape N x A x D, where N is the number
        of combinations in the design space, A is the number of amino acids in 
        the combination, and D is the dimensionality of the base encoding. This
        array contains the unnormalized MLDE encodings.
        
    Returns
    -------
    normalized_encodings: Numpy array with the same shape as the input, only
        with encodings mean-centered and unit-scaled
    """
    # Get the length of a flattened array
    flat_shape = np.prod(unnormalized_encodings.shape[1:])

    # Flatten the embeddings along the internal axes
    flat_encodings = np.reshape(unnormalized_encodings,
                                 [len(unnormalized_encodings), flat_shape])

    # Mean-center and unit variance scale the embeddings.
    means = flat_encodings.mean(axis=0)
    stds = flat_encodings.std(axis=0)
    # Create a copy of stds
    stds_safe = np.copy(stds)

    # Replace 0s with 1s. This prevents division by zero.
    stds_safe[stds_safe == 0] = 1 

    # Perform the safe division
    normalized_flat_encodings = (flat_encodings - means) / stds_safe

    # Just in case, replace any NaNs that might have slipped through
    normalized_flat_encodings = np.nan_to_num(normalized_flat_encodings, nan=0.0)
    
    # Reshape the normalized embeddings back to the original form
    normalized_encodings = np.reshape(normalized_flat_encodings,
                                      unnormalized_encodings.shape)
    
    return normalized_encodings


def encode(data, wt_sequence, model_scope_mutations):
    """
    Parameters
    ----------
    data: dataframe with amino acid combination in one column and corresponding activity values in another column
    positions: number of positions being combinatorially explored
    
    Returns
    -------
    encodings_dictionary: flattened and normalized encodings; dictionary key is type of encoding and values are encoded positions
    """
    
    # Import data
    if data.empty == False:
        sequences = data['AminoAcid'].values
    mutation_regex = re.compile(r"([A-Z])(\d+)([A-Z])")

    ALL_AAS_DICT = {aa: i for i, aa in enumerate(ALL_AAS)}
    
    # Build a dictionary that links the identity of a combination to that combination's index in an encoding array, and vice versa.
    all_combos = model_scope_mutations
    combo_to_index = {"".join(combo): i for i, combo in enumerate(all_combos)}
    sequence_length = len(wt_sequence)

    ## Onehot encodings
    wt_onehot = np.zeros([sequence_length, 20])
    for i, aa in enumerate(wt_sequence):
        wt_onehot[i, ALL_AAS_DICT[aa]] = 1
    
    onehot_array = np.zeros([len(all_combos), sequence_length, 20])
    for i, combo_string in enumerate(all_combos):
        mutant_encoding = np.copy(wt_onehot)
        match = mutation_regex.match(combo_string)
        if not match:
            print(f"Warning: Could not parse {combo_string}")
            onehot_array[i] = mutant_encoding
            continue
        orig_aa, pos_str, new_aa = match.groups()
        pos_0_indexed = int(pos_str) - 1

        mutant_encoding[pos_0_indexed, ALL_AAS_DICT[orig_aa]] = 0
        mutant_encoding[pos_0_indexed, ALL_AAS_DICT[new_aa]] = 1

        onehot_array[i] = mutant_encoding
    
    if data.empty == True: 
        o_encode_array = onehot_array
    else:
        o_encode_array = []
        for combo in sequences:
            index = combo_to_index[combo]
            encoding = onehot_array[index]
            o_encode_array.append(encoding)
        o_encode_array = np.array(o_encode_array)
    
    flat_length = np.prod(o_encode_array.shape[1:])
    x_onehot = np.reshape(o_encode_array, [len(o_encode_array), flat_length])
    
    
    ## Georgiev encodings
    wt_georgiev = np.empty([sequence_length, 19])
    for i, aa in enumerate(wt_sequence):
        wt_georgiev[i] = [param[aa] for param in GEORGIEV_PARAMETERS]
    
    unnorm_georgiev = np.empty([len(all_combos), sequence_length, 19])

    for i, combo_string in enumerate(all_combos):
        mutant_encoding = np.copy(wt_georgiev)
        match = mutation_regex.match(combo_string)
        if not match:
            unnorm_georgiev[i] = mutant_encoding
            continue
        _, pos_str, new_aa = match.groups()
        pos_0_indexed = int(pos_str) - 1

        mutant_encoding[pos_0_indexed] = [param[new_aa] for param in GEORGIEV_PARAMETERS]
        unnorm_georgiev[i] = mutant_encoding

    norm_georgiev = normalize_encodings(unnorm_georgiev)
    
    if data.empty == True: 
        g_encode_array = norm_georgiev
    else:
        g_encode_array = []
        for combo in sequences:
            index = combo_to_index[combo]
            g_encoding = norm_georgiev[index]
            g_encode_array.append(g_encoding)
        g_encode_array = np.array(g_encode_array)
    
    flat_length = np.prod(g_encode_array.shape[1:])
    x_georgiev = np.reshape(g_encode_array, [len(g_encode_array), flat_length])
    
    
    ## ZScale encodings
    wt_zscale = np.empty([sequence_length, 5])
    for i, ass in enumerate(wt_sequence):
        wt_zscale[i] = ZSCALE[aa]
    
    unnorm_zscale = np.empty([len(all_combos), sequence_length, 5])

    for i, combo_string in enumerate(all_combos):
        mutant_encoding = np.copy(wt_zscale)
        match = mutation_regex.match(combo_string)
        if not match:
            unnorm_zscale[i] = mutant_encoding
            continue

        _, pos_str, new_aa = match.groups()
        pos_0_indexed = int(pos_str) - 1

        mutant_encoding[pos_0_indexed] = ZSCALE[new_aa]

        unnorm_zscale[i] = mutant_encoding

    norm_zscale = normalize_encodings(unnorm_zscale)    
    
    if data.empty == True: 
        z_encode_array = norm_zscale
    else:
        z_encode_array = []
        for combo in sequences:
            index = combo_to_index[combo]
            z_encoding = norm_zscale[index]
            z_encode_array.append(z_encoding)
        z_encode_array = np.array(z_encode_array)
    
    flat_length = np.prod(z_encode_array.shape[1:])
    x_zscale = np.reshape(z_encode_array, [len(z_encode_array), flat_length])
    
    
    ## VHSE Encodings
    wt_vhse = np.empty([sequence_length, 8])
    for i, aa in enumerate(wt_sequence):
        wt_vhse[i] = VHSE[aa]

    unnorm_vhse = np.empty([len(all_combos), sequence_length, 8])

    for i, combo_string in enumerate(all_combos):
        mutant_encoding = np.copy(wt_vhse)

        match = mutation_regex.match(combo_string)
        if not match:
            unnorm_vhse[i] = mutant_encoding
            continue

        _, pos_str, new_aa = match.groups()
        pos_0_indexed = int(pos_str) - 1

        mutant_encoding[pos_0_indexed] = VHSE[new_aa]

        unnorm_vhse[i] = mutant_encoding

    norm_vhse = normalize_encodings(unnorm_vhse)    
    
    if data.empty == True: 
        v_encode_array = norm_vhse
    else:
        v_encode_array = []
        for combo in sequences:
            index = combo_to_index[combo]
            v_encoding = norm_vhse[index]
            v_encode_array.append(v_encoding)
        v_encode_array = np.array(v_encode_array)
    
    flat_length = np.prod(v_encode_array.shape[1:])
    x_vhse = np.reshape(v_encode_array, [len(v_encode_array), flat_length])
    
    
    ## Physical Descriptors Encodings
    wt_pd = np.empty([sequence_length, 3])
    for i, aa in enumerate(wt_sequence):
        wt_pd[i] = PYHSICAL_DESCRIPTORS[aa]

    unnorm_pd = np.empty([len(all_combos), sequence_length, 3])

    for i, combo_string in enumerate(all_combos):
        mutant_encoding = np.copy(wt_pd)
        match = mutation_regex.match(combo_string)
        if not match:
            unnorm_pd[i] = mutant_encoding
            continue

        _, pos_str, new_aa = match.groups()
        pos_0_indexed = int(pos_str) - 1

        mutant_encoding[pos_0_indexed] = PYHSICAL_DESCRIPTORS[new_aa]
        unnorm_pd[i] = mutant_encoding

    norm_pd = normalize_encodings(unnorm_pd)    
    
    if data.empty == True: 
        p_encode_array = norm_pd
    else:
        p_encode_array = []
        for combo in sequences:
            index = combo_to_index[combo]
            p_encoding = norm_pd[index]
            p_encode_array.append(p_encoding)
        p_encode_array = np.array(p_encode_array)
    
    flat_length = np.prod(p_encode_array.shape[1:])
    x_pd = np.reshape(p_encode_array, [len(p_encode_array), flat_length])
    
    encoding_dict = {'One Hot':x_onehot, 'Georgiev':x_georgiev, 'ZScales':x_zscale, 'VHSE':x_vhse, 'Physical Descriptors':x_pd}
    
    return encoding_dict


def normalize_data(data, normalization_type):
    """
    Parameters
    ----------
    data: dataframe with amino acid combination in one column and corresponding activity values in another column
    normalization_type: str of desired normalizations strategy ('Max-min', 'Standardization', or 'None')
    
    Returns
    -------
    normalized_data: list of normalized y-values
    """
    # Import Data
    activity = data['Activity']

    if normalization_type == 'Max-min':
        max_min_norm = []
        for i in activity:
            x = (i - activity.min()) / (activity.max() - activity.min())
            max_min_norm.append(x)
        normalized_data = max_min_norm
    
    elif normalization_type == 'Standardization':
        standardization = []
        for i in activity:
            x = (i - activity.mean()) / activity.std()
            standardization.append(x)
        normalized_data = standardization
    
    elif normalization_type == 'None':
        normalized_data = activity

    return normalized_data

    
