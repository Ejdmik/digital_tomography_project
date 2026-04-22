import numpy as np
import time
import os
import statistics

from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose3
from pysat.card import CardEnc
from utils import *




def build_instance(m, n, p, encoding, density):
    vpool = IDPool()
    cnf = CNF()

    bitmap = random_bitmap3d(m,n,p,density)

    xy = get_axis_vectors(bitmap, 2)
    xz = get_axis_vectors(bitmap, 1)
    yz = get_axis_vectors(bitmap, 0)



    # encodings
    encode_xy(cnf, vpool, xy, encoding, p)
    encode_xz(cnf, vpool, xz, encoding, n)
    encode_yz(cnf, vpool, yz, encoding, m)

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
    outfile = f"difficulty_sat_6x6x6_dens_tot_{job_id}.txt"

    m, n, p = 6, 6, 6
    runs = 1000
    encoding = 2 # seqcounter

    times = []

    for _ in range(runs):
        cnf = build_instance(m, n, p, encoding, job_id)

        t = solve_instance(cnf)

        times.append(t)


    # SAVE RESULTS
    with open(outfile, "w") as f:
        f.write(f"density={job_id}, avg_time={statistics.mean(times)}, med_time={statistics.median(times)}\n")


if __name__ == "__main__":
    main()
