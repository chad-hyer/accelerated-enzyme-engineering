#!/bin/bash
#
# --- Slurm Job Details ---
#SBATCH --job-name=tiling_worker
#SBATCH -c 1
#SBATCH --mem=1500M
#SBATCH --time=0-00:02:00
#
# All #SBATCH directives are above this line.
# --- Shell Script Starts Here ---

echo "--- Starting Slurm Task $SLURM_ARRAY_TASK_ID ---"
date

# --- Arguments passed from sbatch command ---
IT_DIR=$1
MUTATION_LIST_PATH=$2

echo "Iteration Dir: $IT_DIR"
echo "Mutation List: $MUTATION_LIST_PATH"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not defined."
    exit 1
fi

echo "Passing Task ID $SLURM_ARRAY_TASK_ID to Python."

# --- Activate Environment ---
ml python/3.12.1
source /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/tiling_env/bin/activate

# Set thread counts for safety
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Run the Python Worker Script ---
# We pass the Task ID as the 3rd argument
python3 /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/Array_edits/process_one_mutation.py \
    $IT_DIR \
    $MUTATION_LIST_PATH \
    $SLURM_ARRAY_TASK_ID

echo "--- Finished Slurm Task $SLURM_ARRAY_TASK_ID ---"
date