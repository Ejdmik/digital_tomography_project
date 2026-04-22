import numpy as np
import sys
import os
from solver_exact import *
from utils import *



if __name__ == "__main__":
    job_id = int(sys.argv[1])
    seed = 1234 + job_id
    np.random.seed(seed)
    outfile = f"exact_sat_15x15_dens_{job_id}.txt"

    m, n = 15, 15
    runs = 1000

    results = []

    print(f"Running density {job_id}")

    for i in range(runs):
        bitmap = random_bitmap(m, n, job_id)

        row = get_r_vector(bitmap).tolist()
        col = get_c_vector(bitmap).tolist()

        number, _ = count(row, col)
        results.append(number)

    median = np.median(results)

    with open(outfile, "w") as f:
        f.write(f"density={job_id}, median={median}\n")

    print("Finished.")



    # for each density it computes median number of solutions