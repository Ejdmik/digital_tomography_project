#!/bin/bash
#SBATCH --job-name=exact
#SBATCH --partition=cpufast
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --array=0-36

echo "Running on $(hostname)"
echo "Starting at $(date)"

source ~/venv/bin/activate

echo "Task ID: $SLURM_ARRAY_TASK_ID"

python run_exact_solver.py $SLURM_ARRAY_TASK_ID

echo "Finished at $(date)"