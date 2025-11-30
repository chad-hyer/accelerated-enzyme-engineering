import re
import numpy as np
import pandas as pd

ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

# Compile the regex for parsing mutations
MUTATION_REGEX = re.compile(r"([A-Z])(\d+)([A-Z])")

def create_error_matrix(dataframe):
    """
    Creates an error matrix (Residue x AA) from a dataframe with 
    'AminoAcid', 'Error', and 'Class' columns.

    Rules:
    - 'train' class mutations are set to 0.0
    - Wild-type mutations (e.g., "A10A") are set to 0.0
    - 'test' class mutations are set to their 'Error' value
    - All other possible mutations (not in the dataframe) are set to np.nan
    """
    
    # Work on a copy to avoid changing the original dataframe
    df = dataframe.copy()

    # --- 1. Parse Mutation Data ---
    parsed_data = df['AminoAcid'].astype(str).str.extract(MUTATION_REGEX)
    df['orig_aa'] = parsed_data[0]
    df['residue'] = parsed_data[1]
    df['new_aa'] = parsed_data[2]

    # Drop any rows that didn't parse correctly and convert residue to int
    df = df.dropna(subset=['residue', 'new_aa'])
    df['residue'] = df['residue'].astype(int)

    # --- 2. Apply Filling Logic ---
    
    # Create a boolean column for "is_wt"
    df['is_wt'] = (df['orig_aa'] == df['new_aa'])
    
    # Create the final value column based on your rules
    # We use np.where for this:
    # 1. If 'Class' is 'train' OR 'is_wt' is True, set value to 0.0
    # 2. Otherwise, use the value from the 'Error' column
    df['value_to_fill'] = np.where(
        (df['Class'] == 'train') | (df['is_wt']), 
        0.0, 
        df['Error']
    )

    # --- 3. Create the Matrix ---
    
    # Pivot the dataframe to create the matrix
    # This automatically puts NaN where no data exists
    error_matrix_sparse = df.pivot(
        index='residue', 
        columns='new_aa', 
        values='value_to_fill'
    )

    # --- 4. Format the Final Matrix ---
    
    # Find the max residue number to create a full index
    max_residue = df['residue'].max()
    all_residues = range(1, max_residue + 1)
    
    # Reindex to ensure all residues (1 to max) and all 20 AAs are present
    final_error_matrix = error_matrix_sparse.reindex(
        index=all_residues, 
        columns=ALL_AAS
    )
    
    return final_error_matrix