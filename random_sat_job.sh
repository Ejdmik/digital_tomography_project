#!/bin/bash
#SBATCH --job-name=random_sat
#SBATCH --partition=cpulong
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --array=0-225

echo "Running on $(hostname)"
echo "Starting at $(date)"

source venv/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"

python random_sat_experiment.py

echo "Finished at $(date)"
