#!/bin/bash
#SBATCH --job-name=ganak_array
#SBATCH --partition=cpulong
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --array=0-216

echo "Running on $(hostname), task $SLURM_ARRAY_TASK_ID"
echo "Start $(date)"

source venv/bin/activate

python run_ganak3d.py

echo "Done $(date)"