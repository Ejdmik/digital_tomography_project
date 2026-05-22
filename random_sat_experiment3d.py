import numpy as np
import time
import os

from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose3
from pysat.card import CardEnc
from utils import *




def build_instance(m, n, p, encoding, density):
    # generator gen: 0 = random_axis_vector,  1  = random_axis_vector2
    vpool = IDPool()
    cnf = CNF()

    xy_gen = 1
    xz_gen = 1
    yz_gen = 1

    # random projections
    if xy_gen == 0:
        xy = random_axis_vector(m, n, density, p)
    else:
        xy = random_axis_vector2(m, n, density, p)

    if xz_gen == 0:
        xz = random_axis_vector(m, p, density, n)
    else:
        xz = random_axis_vector2(m, p, density, n)

    if yz_gen == 0:
        yz = random_axis_vector(n, p, density, m)
    else:
        yz = random_axis_vector2(n, p, density, m)
    

    encode_xy(cnf, vpool, xy, encoding, p)
    encode_xz(cnf, vpool, xz, encoding, n)
    encode_yz(cnf, vpool, yz, encoding, m)

    return cnf



def solve_instance(cnf):
    solver = Glucose3(bootstrap_with=cnf.clauses)

    sat = solver.solve()

    solver.delete()
    return sat



def main():
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    seed = 1234 + job_id
    np.random.seed(seed)
    outfile = f"random_sat3d_6x6x6_dens_{job_id}.txt"

    m, n, p = 6, 6, 6
    runs = 1000
    encoding = EncType.seqcounter

    sat_count = 0

    for i in range(runs):
        cnf = build_instance(m, n, p, encoding, job_id)

        sat = solve_instance(cnf)

        sat_count += int(sat)


    # SAVE RESULTS
    with open(outfile, "w") as f:
        f.write(f"density={job_id}, sat={sat_count}\n")


if __name__ == "__main__":
    main()
