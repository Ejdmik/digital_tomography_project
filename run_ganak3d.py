import numpy as np
import random
import os
import time
import statistics
import subprocess
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from utils import *

m, n, p = 6, 6, 6
RUNS = 100
GANAK_PATH = "./ganak_executable"
ENCODING = 1


def write_cnf_with_header(cnf, filepath, m, n, p):
    show_vars = " ".join(str(i) for i in range(1, m * n * p + 1))
    
    with open(filepath, "w") as f:
        f.write("c t pmc\n")
        f.write(f"c p show {show_vars} 0\n")
        f.write(f"p cnf {cnf.nv} {len(cnf.clauses)}\n")
        for clause in cnf.clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")



def build_instance(m, n, p, encoding, density):
    vpool = IDPool()
    cnf = CNF()

    # create variables
    for i in range(1, m+1):
        for j in range(1, n+1):
            for k in range(1, p+1):
                vpool.id((i, j, k))

    bitmap = random_bitmap3d(m, n, p, density)

    # projections
    P_xy = np.sum(bitmap, axis=2)  # sum over z
    P_xz = np.sum(bitmap, axis=1)  # sum over y
    P_yz = np.sum(bitmap, axis=0)  # sum over x


    # encode constraints
    encode_xy(cnf, vpool, P_xy, encoding, p)
    encode_xz(cnf, vpool, P_xz, encoding, n)
    encode_yz(cnf, vpool, P_yz, encoding, m)

    encode_plane_diagonals_from_bitmap(cnf, vpool, bitmap, m, n, p, encoding)

    return cnf


def run_ganak(cnf_file):
    result = subprocess.run(
        [GANAK_PATH, cnf_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    output = result.stdout
    count = None
    solve_time = None

    for line in output.splitlines():
        if "exact arb int" in line:
            count = int(line.split()[-1])
        if "Total time" in line:
            solve_time = float(line.split()[-1])

    return count, solve_time


def main():
    density = int(os.environ["SLURM_ARRAY_TASK_ID"])

    random.seed(1234 + density)
    np.random.seed(1234 + density)

    counts = []
    times = []

    for i in range(RUNS):
        cnf = build_instance(m, n, p, ENCODING, density)

        cnf_file = f"tmp_3d_xyz_{density}_{i}.cnf"
        write_cnf_with_header(cnf, cnf_file, m, n, p)

        try:
            count, t = run_ganak(cnf_file)
        finally:
            os.remove(cnf_file)
        
        if count is None:
            print(f"Warning: ganak failed on run {i}, skipping")
            continue

        print(f"density={density}, count={count}, time={t}", flush=True)

        counts.append(count)
        times.append(t)

    with open(f"ganak3d_results_{density}.txt", "w") as f:
        f.write(f"density={density}\n")
        f.write(f"median_count={statistics.median(counts)}\n")
        f.write(f"median_time={statistics.median(times)}\n")


if __name__ == "__main__":
    main()