#!/bin/bash
#SBATCH --job-name=ganak
#SBATCH --partition=cpulong
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=4G

echo "Running on $(hostname)"
echo "Start $(date)"

CNF="dt_small_bird_rcab.cnf"
OUT="ganak_small_bird_rcab_results.txt"

/usr/bin/time -f "TIME %e" ./ganak $CNF 2>&1 \
| grep -E "exact arb int|TIME" \
> $OUT

echo "Done $(date)"