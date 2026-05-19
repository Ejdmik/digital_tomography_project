import numpy as np
import sys
import os
from solver_exact import *
from utils import *
import time



if __name__ == "__main__":
    job_id = int(sys.argv[1])
    seed = 1234 + job_id
    np.random.seed(seed)
    outfile = f"exact_sat_6x6_dens_{job_id}.txt"

    m, n = 6, 6
    runs = 100

    results = []
    times = []

    print(f"Running density {job_id}")

    for i in range(runs):
        bitmap = random_bitmap(m, n, job_id)

        row = get_r_vector(bitmap).tolist()
        col = get_c_vector(bitmap).tolist()

        start = time.perf_counter()
        number, _ = count(row, col)
        t = time.perf_counter() - start

        results.append(number)
        times.append(t)


    with open(outfile, "w") as f:
        f.write(f"density={job_id}\n")
        f.write(f"median_count={statistics.median(results)}\n")
        f.write(f"median_time={statistics.median(times)}\n")

    print("Finished.")



    # for each density it computes median number of solutions