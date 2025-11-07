#!/bin/bash

# --- Job Details ---
#SBATCH --job-name=tiling_conductor  # Name for this master job
#SBATCH --time=2-00:00:00            # Max walltime: 7 days (adjust as needed) # --partition=long             # Run on a partition with a long walltime

# --- Resource Allocation ---
# This master script is serial. It just launches and waits for other jobs.
# It only needs 1 CPU and a moderate amount of memory to
# process the results at the end of each iteration.
#SBATCH -N 1                         # 1 Node
#SBATCH -c 1                         # 1 CPU core
#SBATCH --mem=16G                    # 16GB RAM (for loading/processing results)

# --- Logging and Notifications ---
# These logs will capture the print() statements from Tiling_HPC_Conductor.py
#SBATCH --output=conductor_%j.out    # Conductor's main log file (%j = job ID)
#SBATCH --error=conductor_%j.err     # Conductor's error file
#SBATCH --mail-type=ALL              # Email on start, end, and fail
#SBATCH --mail-user=chadhyer@stanford.edu # Your email (from your original script)

# --- Environment Setup ---
echo "--- Starting Tiling Conductor ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
date

# Load the Python module
ml python/3.12.1

# Activate the virtual environment
source /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/tiling_env/bin/activate

# Set thread counts to 1 to prevent numpy/sklearn from over-threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Run the Conductor Python Script ---
# This one python script will run for the entire duration of the experiment,
# submitting new job arrays for each iteration.
python3 /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/Array_edits/Tiling_HPC_Conductor.py

echo "--- Tiling Conductor Finished ---"
date