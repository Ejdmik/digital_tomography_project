import numpy as np
import random
import os
import time
import statistics
import subprocess
from pysat.formula import CNF, IDPool
from utils import *

m, n = 15, 15
RUNS = 100
ENCODING = 1

def write_cnf_with_header(cnf, filepath, m, n):
    show_vars = " ".join(str(i) for i in range(1, m * n + 1))
    
    with open(filepath, "w") as f:
        # Write pmc headers first
        f.write("c t pmc\n")
        f.write(f"c p show {show_vars} 0\n")
        # Write p cnf line
        num_vars = cnf.nv
        num_clauses = len(cnf.clauses)
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        # Write clauses
        for clause in cnf.clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def build_instance(m, n, encoding, density):
    vpool = IDPool()
    cnf = CNF()

    for i in range(1, m+1):
        for j in range(1, n+1):
            vpool.id((i, j))

    bitmap = random_bitmap(m,n,density)

    rows = get_r_vector(bitmap)
    cols = get_c_vector(bitmap)
    a = get_a_vector(bitmap)
    b = get_b_vector(bitmap)
    a5 = get_a_with_slope_vector(bitmap, 5)
    b5 = get_b_with_slope_vector(bitmap, 5)
    f = get_frames_vector(bitmap)


    encode_rows(cnf, vpool, rows, encoding, n)
    encode_cols(cnf, vpool, cols, encoding, m)
    encode_a_diagonals(cnf, vpool, a, encoding, m, n)
    encode_b_diagonals(cnf, vpool, b, encoding, m, n)
    #encode_a_slope_diagonals(cnf, vpool, a5, 5, encoding, m, n)
    #encode_b_slope_diagonals(cnf, vpool, b5, 5, encoding, m, n)
    #encode_frames(cnf, vpool, f, encoding, m, n)


    return cnf


def run_approxmc(cnf_file):

    result = subprocess.run(
        ["./approxmc", cnf_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    output = result.stdout

    count = None
    solve_time = None

    for line in output.splitlines():
        line = line.strip()

        # Parse count
        if line.startswith("s mc"):
            try:
                count = int(line.split()[2])
            except (IndexError, ValueError):
                pass

        # Parse total time
        if "[appmc+arjun] Total time:" in line:
            try:
                solve_time = float(line.split(":")[-1].strip())
            except (IndexError, ValueError):
                pass

    return count, solve_time


def main():
    density = int(os.environ["SLURM_ARRAY_TASK_ID"])

    random.seed(1234 + density)
    np.random.seed(1234 + density)

    counts = []
    times = []

    for i in range(RUNS):
        cnf = build_instance(m, n, ENCODING, density)

        cnf_file = f"approxmc_tmp_{density}_{i}.cnf"
        write_cnf_with_header(cnf, cnf_file, m, n)

        try:
            count, t = run_approxmc(cnf_file)
        finally:
            os.remove(cnf_file)
        
        if count is None:
            print(f"Warning: approxmc failed on run {i}, skipping")
            continue

        print(f"count={count} time={t}", flush=True)
        counts.append(count)
        times.append(t)

    with open(f"approxmc_results_{density}.txt", "w") as f:
        f.write(f"density={density}\n")
        f.write(f"median_count={statistics.median(counts)}\n")
        f.write(f"median_time={statistics.median(times)}\n")


if __name__ == "__main__":
    main()
