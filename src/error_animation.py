# ChimeraX Python script to create a movie from a series of .defattr files
#
# How to use:
# 1. Open your protein model in ChimeraX.
# 2. Open this script file in ChimeraX (e.g., via the Command Line: runscript "path/to/this/script.py")
#    or paste its content into the Command Line (prefixed with "py:").
# 3. Make sure to update the variables in the "--- CONFIGURATION ---" section below.

import os, time
import glob
import re
import pandas as pd
from chimerax.core.commands import run

# --- CONFIGURATION ---
# PLEASE UPDATE THESE PATHS

# 1. Set the directory where your .defattr files are located
#    The r"..." (raw string) is important for Windows paths.
DEFATTR_DIR = r"D:\Downloads\Files to work with\ChimeraX_defattr"

# 2. Set the file pattern to match your files
FILE_PATTERN = "it_*.defattr"

# 3. Set the model ID of your protein (e.g., #1).
#    Check the ChimeraX interface to see what your model's ID is.
MODEL_ID = "#1"

# 4. Set your desired coloring command
#    We use an f-string to insert the MODEL_ID.
COLOR_COMMAND = f"color byattribute median_error palette 0,blue:0.5,white:1,red"

# 5. Set the full path for the output movie file
OUTPUT_MOVIE_PATH = r"D:\Downloads\Files to work with\protein_error_animation.mp4"

# --- END OF CONFIGURATION ---


def create_movie(session):
    """
    Finds, sorts, and processes .defattr files to create a movie.
    """
    print(f"--- Starting movie creation ---")
    df = pd.read_csv("D:/Downloads/Files to work with/ChimeraX_defattr/perfect_tiling_path.tsv",delimiter='\t')
    df['Residue Number'] = df['Best_Mutation'].str[1:-1].astype(int)
    run(session, "movie record")
    # --- 1. Find all attribute files ---
    file_search_path = os.path.join(DEFATTR_DIR, FILE_PATTERN)
    print(f"Searching for files in: {file_search_path}")
    defattr_files = glob.glob(file_search_path)
    
    if not defattr_files:
        print(f"Error: No files found matching pattern '{FILE_PATTERN}' in directory '{DEFATTR_DIR}'.")
        print("Please check your DEFATTR_DIR and FILE_PATTERN variables.")
        return

    print(f"Found {len(defattr_files)} attribute files.")

    # --- 2. Sort files numerically ---
    # This helper function extracts the frame number from the filename
    def get_frame_number(filepath):
        filename = os.path.basename(filepath)
        # This regex matches 'it_' followed by one or more digits (\d+)
        match = re.search(r'it_(\d+)\.defattr', filename)
        if match:
            return int(match.group(1))
        # Return a default value if no match (shouldn't happen with your pattern)
        return -1 

    try:
        sorted_files = sorted(defattr_files, key=get_frame_number)
        print("Successfully sorted files numerically.")
    except Exception as e:
        print(f"Error sorting files: {e}")
        print("Files found:", defattr_files)
        return

    # --- 3. Loop and create movie frames ---
    print("Adding frames to movie...")
    run(session, "roll")
    #time.sleep(1)
    offeset = 0
    for i, attr_file in enumerate(sorted_files):
        run(session, "2dlabels delete")
        row = df.loc[i]
        mut = row['Best_Mutation']
        rn = row['Residue Number']
        it = row['Iteration']
        run(session, f"2dlabels text 'Iteration: {it}' xpos 0.5 ypos 0.9")
        print(f"Processing: {os.path.basename(attr_file)}")
        
        # Create a Windows-safe path for the ChimeraX command
        # We must escape the backslashes for the ChimeraX command interpreter
        safe_attr_path = attr_file.replace('\\', '\\\\')
        
        # a. Load the attribute file
        run(session, f"open \"{safe_attr_path}\"")
        run(session, f'color :{rn} lime')
        run(session, "wait 1")
        
        # b. Apply the coloring
        run(session, COLOR_COMMAND)
        
        # c. Add the current view as a movie frame
        #

    # --- 4. Encode and save the movie ---
    #run(session, "movie stop")
    run(session, "stop")
    print(f"Encoding movie... this may take a moment.")
    safe_output_path = OUTPUT_MOVIE_PATH.replace('\\', '\\\\')
    
    # You can add more options here, e.g., framerate:
    run(session, f"movie encode \"{safe_output_path}\" framerate 30")
    #run(session, f"movie encode \"{safe_output_path}\"")
    
    print(f"--- Movie saved to: {OUTPUT_MOVIE_PATH} ---")


# --- Run the main function ---
# This is the entry point when ChimeraX executes the script
try:
    create_movie(session)
except NameError:
    print("\n--- This script must be run from within ChimeraX ---")
    print("1. Open your model in ChimeraX.")
    print("2. Open this script file (create_attribute_movie.py).")
    print("3. In the ChimeraX command line, type: runscript \"path\\to\\create_attribute_movie.py\"")
    
