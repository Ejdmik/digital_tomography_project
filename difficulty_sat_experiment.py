import numpy as np
import time
import os
import statistics

from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose3
from pysat.card import CardEnc
from utils import *




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


    # encodings
    encode_rows(cnf, vpool, rows, encoding, n)
    encode_cols(cnf, vpool, cols, encoding, m)
    #encode_a_diagonals(cnf, vpool, a, encoding, m, n)
    #encode_b_diagonals(cnf, vpool, b, encoding, m, n)

    return cnf



def solve_instance(cnf):
    solver = Glucose3(bootstrap_with=cnf.clauses)


    start = time.perf_counter()
    solver.solve()
    t = time.perf_counter() - start

    solver.delete()
    return t



def main():
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = 1234 + job_id
    np.random.seed(seed)
    outfile = f"difficulty_sat_15x15_rc_dens_{5*(job_id + 1)}.txt"

    m, n = 15, 15
    runs = 1000
    encoding = 1 # seqcounter

    times = []

    for _ in range(runs):
        cnf = build_instance(m, n, encoding, 5*(job_id + 1))

        t = solve_instance(cnf)

        times.append(t)


    # SAVE RESULTS
    with open(outfile, "w") as f:
        f.write(f"density={5*(job_id + 1)}, avg_time={statistics.mean(times)}, med_time={statistics.median(times)}\n")


if __name__ == "__main__":
    main()