#!/bin/bash
#SBATCH --job-name=exact
#SBATCH --output=logs/exact_%A_%a.out
#SBATCH --partition=cpulong
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --array=0-6

echo "Running on $(hostname)"
echo "Starting at $(date)"

source ~/venv/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"

python run_exact_solver.py $SLURM_ARRAY_TASK_ID

echo "Finished at $(date)"