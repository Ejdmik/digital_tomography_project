#!/bin/bash
#SBATCH --job-name=d4v2_15x15_rcab
#SBATCH --time=2-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpulong
#SBATCH --array=0-225

module load GCCcore/13.2.0
module load GCC/13.2.0
module load Ninja/1.11.1-GCCcore-13.2.0
module load CMake/3.27.6-GCCcore-13.2.0
module load Boost/1.83.0-GCC-13.2.0
module load GMP/6.3.0-GCCcore-13.2.0

echo "Running on $(hostname), task $SLURM_ARRAY_TASK_ID"
echo "Start $(date)"

source venv/bin/activate

python3 experiment_d4v2.py

echo "Job finished for density=$SLURM_ARRAY_TASK_ID"