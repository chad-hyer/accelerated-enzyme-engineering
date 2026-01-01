#!/bin/bash
#SBATCH --job-name=copy_files
#SBATCH --output=copy_log.out
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# 1. Create the destination directories if they don't exist
mkdir -p predictions
mkdir -p errors

echo "Starting copy process..."

# 2. Loop through all directories matching the pattern 'it_*'
# The loop variable 'dir' will hold the name (e.g., "it_5", "it_100")
for dir in it_*; do
    if [ -d "$dir" ]; then
        # Construct the specific filenames based on the directory name
        # Example: if dir is "it_5", file is "prediction_it_5.csv"
        pred_file="${dir}/prediction_${dir}.csv"
        err_file="${dir}/median_error_${dir}.csv"

        # 3. Copy prediction file if it exists
        if [ -f "$pred_file" ]; then
            cp "$pred_file" predictions/
        fi

        # 4. Copy error file if it exists
        if [ -f "$err_file" ]; then
            cp "$err_file" errors/
        fi
    fi
done

echo "Copy process complete."