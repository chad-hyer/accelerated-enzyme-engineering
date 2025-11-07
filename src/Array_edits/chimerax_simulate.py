# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:47:24 2025

@author: hyerc
"""

import pandas as pd
import numpy as np
import re
import os

def create_chimera_attribute_file(iteration_data_file, output_attr_file, error_column_name='Error'):
    """
    Processes a mutation error file from a single iteration and creates
    a ChimeraX attribute file for coloring by median residue error.

    Args:
        iteration_data_file (str): Path to the CSV file for one iteration.
                                   (e.g., 'iteration_1.csv')
        output_attr_file (str): Path to save the new attribute file.
                                (e.g., 'attr_files/iter_1.attr')
        error_column_name (str): The name of the column containing your error metric.
    """
    
    # --- 1. Load the data ---
    try:
        data = pd.read_csv(iteration_data_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {iteration_data_file}")
        return
    except Exception as e:
        print(f"Error loading {iteration_data_file}: {e}")
        return

    if error_column_name not in data.columns:
        print(f"Error: Column '{error_column_name}' not found in {iteration_data_file}")
        return
    
    # --- 2. Parse Residue Number ---
    
    # This regex is robust for mutations like "A10V"
    mutation_regex = re.compile(r"[A-Z](\d+)[A-Z]") 
    
    def get_residue_num(mutation_str):
        match = mutation_regex.match(str(mutation_str))
        if match:
            return int(match.group(1)) # Return the number (e.g., 10)
        return None

    data['residue'] = data['AminoAcid'].apply(get_residue_num)
    
    # Drop any rows that couldn't be parsed (e.g., WT, or different format)
    data = data.dropna(subset=['residue'])
    data['residue'] = data['residue'].astype(int)

    # --- 3. Calculate Median Error Per Residue ---
    median_error_by_residue = data.groupby('residue')[error_column_name].median()
    
    # --- 4. Write the ChimeraX Attribute File ---
    with open(output_attr_file, 'w') as f:
        # Header
        f.write("attribute: median_error\n")
        f.write("recipient: residues\n")
        f.write("# Description: Median error for this iteration\n")
        
        # Data
        for residue_num, median_error in median_error_by_residue.items():
            # Format: ":<residue_number> <value>"
            f.write(f"\t:{residue_num}\t{median_error:.6f}\n")

    print(f"Successfully created {output_attr_file}")

# --- Example of how to run this for all your iterations ---
"""
# Create a directory to hold the attribute files
os.makedirs('chimera_attr_files', exist_ok=True)

# Define the number of iterations you have
TOTAL_ITERATIONS = 5000 # Change this to your total number

print("Starting to generate attribute files...")

for i in range(1, TOTAL_ITERATIONS + 1):
    
    # Define your input and output file names
    # (Assumes your files are named 'iteration_1.csv', 'iteration_2.csv', etc.)
    input_file = f'path/to/your/data/iteration_{i}.csv'
    output_file = f'chimera_attr_files/iter_{i:04d}.defattr' # e.g., iter_0001.attr
    
    # Run the processing function
    create_chimera_attribute_file(input_file, output_file, error_column_name='Error') # Use your error column name

print("All attribute files generated.")
"""