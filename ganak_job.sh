#!/bin/bash
#SBATCH --job-name=ganak
#SBATCH --partition=cpufast
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=4G

echo "Running on $(hostname)"
echo "Start $(date)"

CNF="dt_15x15_rcab.cnf"
OUT="ganak_15x15_rcab_results.txt"

/usr/bin/time -f "TIME %e" ./ganak_executable $CNF 2>&1 \
| grep -E "exact arb int|TIME" \
> $OUT

echo "Done $(date)"