#!/bin/bash
#SBATCH --job-name=tomography
#SBATCH --output=logs_%A_%a.txt
#SBATCH --partition=cpulong
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --array=0-18

echo "Running on $(hostname)"
echo "Starting at $(date)"

source ~/venv/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"

python run_experiment.py $SLURM_ARRAY_TASK_ID

echo "Finished at $(date)"