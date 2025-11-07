#!/bin/bash
#SBATCH --job-name=tiling_worker
#SBATCH -c 1                # Each worker job is single-threaded
#SBATCH --mem=1500M            # Memory for ONE job. Adjust as needed.
#SBATCH --time=0-00:30:00   # Walltime for ONE job (e.g., 30 min).
#
# --- IMPORTANT: Define log output paths ---
# We create a log directory for this iteration
# %A is the Job Array ID, %a is the Task ID
IT_NUM=$1
LOG_DIR="/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/slurm_logs/it_${IT_NUM}"
mkdir -p $LOG_DIR
#SBATCH --output=%x-%A-%a.out  # %x=job-name, %A=array-id, %a=task-id
#SBATCH --error=%x-%A-%a.err
#SBATCH --workdir=$LOG_DIR     # Set working directory to the log dir

# --- Arguments passed from sbatch command ---
IT_DIR=$2
MUTATION_LIST_PATH=$3
TRAIN_PATH=$4
TEST_PATH=$5
EC_PATH=$6
MODEL_SCOPE_PATH=$7

echo "--- Starting Slurm Task $SLURM_ARRAY_TASK_ID ---"
echo "Iteration: $IT_NUM"
echo "Iteration Dir: $IT_DIR"

# Get the Nth mutation from the list, where N is the Slurm task ID
MUTATION_TO_TEST=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $MUTATION_LIST_PATH)

if [ -z "$MUTATION_TO_TEST" ]; then
    echo "Error: Could not find mutation for task $SLURM_ARRAY_TASK_ID."
    exit 1
fi

echo "Processing Mutation: $MUTATION_TO_TEST"

# --- Activate Environment ---
ml python/3.12.1
source /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/tiling_env/bin/activate

# Set thread counts for safety (sklearn can sometimes grab too many)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Run the Python Worker Script ---
# Make sure process_one_mutation.py is in your $BASE_DIR or Python path
python3 /scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/Array_edits/process_one_mutation.py \
    $IT_DIR \
    $MUTATION_TO_TEST \
    $TRAIN_PATH \
    $TEST_PATH \
    $EC_PATH \
    $MODEL_SCOPE_PATH

echo "--- Finished Slurm Task $SLURM_ARRAY_TASK_ID ---"