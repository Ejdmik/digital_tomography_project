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
    a5 = get_a_with_slope_vector(bitmap, 5)
    b5 = get_b_with_slope_vector(bitmap, 5)
    f = get_frames_vector(bitmap)


    encode_rows(cnf, vpool, rows, encoding, n)
    encode_cols(cnf, vpool, cols, encoding, m)
    encode_a_diagonals(cnf, vpool, a, encoding, m, n)
    encode_b_diagonals(cnf, vpool, b, encoding, m, n)
    encode_a_slope_diagonals(cnf, vpool, a5, 5, encoding, m, n)
    encode_b_slope_diagonals(cnf, vpool, b5, 5, encoding, m, n)
    encode_frames(cnf, vpool, f, encoding, m, n)

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
    outfile = f"difficulty_sat_15x15_dens_{job_id}.txt"

    m, n = 15, 15
    runs = 1000
    encoding = EncType.seqcounter

    times = []

    for _ in range(runs):
        cnf = build_instance(m, n, encoding, job_id)

        t = solve_instance(cnf)

        times.append(t)
        print(t, flush=True)


    # SAVE RESULTS
    with open(outfile, "w") as f:
        f.write(f"density={job_id}, med_time={statistics.median(times)}\n")


if __name__ == "__main__":
    main()