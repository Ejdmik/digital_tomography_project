import numpy as np
import time
import os

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

    row_gen = 1
    col_gen = 1
    diag_gen = 1
    anti_diag_gen = 1

    # random projections
    if row_gen == 0:
        rows = random_vector(m, density, n)
    elif row_gen == 1:
        rows = random_vector2(m, density, n)
    else:
        rows = random_vector3(m, density, n)

    if col_gen == 0:
        cols = random_vector(n, density, m)
    elif col_gen == 1:
        cols = random_vector2(n, density, m)
    else:
        cols = random_vector3(n, density, m)

    if diag_gen == 0:
        a = random_diag_vector(m, n, density)
    elif diag_gen == 1:
        a = random_diag_vector2(m, n, density)
    else:
        a = random_diag_vector3(m, n, density)

    if anti_diag_gen == 0:
        b = random_diag_vector(m, n, density)
    elif anti_diag_gen == 1:
        b = random_diag_vector2(m, n, density)
    else:
        b = random_diag_vector3(m, n, density)


    # encodings
    encode_rows(cnf, vpool, rows, encoding, n)
    encode_cols(cnf, vpool, cols, encoding, m)
    encode_a_diagonals(cnf, vpool, a, encoding, m, n)
    encode_b_diagonals(cnf, vpool, b, encoding, m, n)

    return cnf



def solve_instance(cnf):
    solver = Glucose3(bootstrap_with=cnf.clauses)

    start = time.perf_counter()
    sat = solver.solve()
    elapsed = time.perf_counter() - start

    solver.delete()
    return sat, elapsed



def main():
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = 1234 + job_id
    np.random.seed(seed)
    outfile = f"random_sat_15x15_dens_{job_id}.txt"

    m, n = 15, 15
    runs = 1000
    encoding = EncType.seqcounter

    sat_times = []
    unsat_times = []

    for i in range(runs):
        cnf = build_instance(m, n, encoding, job_id)
        sat, elapsed = solve_instance(cnf)

        if sat:
            sat_times.append(elapsed)
        else:
            unsat_times.append(elapsed)

    sat_count = len(sat_times)

    # SAVE RESULTS
    with open(outfile, "w") as f:
        f.write(f"density={job_id}\n")
        f.write(f"sat={sat_count}/{runs}\n")
        f.write(f"median_time_sat={statistics.median(sat_times) if sat_times else 'N/A'}\n")
        f.write(f"median_time_unsat={statistics.median(unsat_times) if unsat_times else 'N/A'}\n")


if __name__ == "__main__":
    main()
