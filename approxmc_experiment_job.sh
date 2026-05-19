#!/bin/bash
#SBATCH --job-name=approxmc
#SBATCH --partition=cpulong
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --array=0-225

echo "Running on $(hostname), task $SLURM_ARRAY_TASK_ID"
echo "Start $(date)"

source venv/bin/activate

python run_approxmc.py

echo "Done $(date)"