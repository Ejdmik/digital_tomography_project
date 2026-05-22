# experiment_d4v2.py
import numpy as np
import random
import os
import time
import subprocess
from pysat.formula import CNF, IDPool
from utils import *
from pysat.card import CardEnc

m, n = 15, 15
D4V2_PATH = os.path.expanduser("~/tmp/d4v2/demo/counter/build/counter")
ENCODING = EncType.seqcounter
RUNS = 1000


def write_cnf_for_d4v2(cnf, filepath, m, n):
    show_vars = " ".join(str(i) for i in range(1, m * n + 1))
    num_vars = cnf.nv
    num_clauses = len(cnf.clauses)

    with open(filepath, "w") as f:
        f.write("c t pmc\n")
        f.write(f"c p show {show_vars} 0\n")
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in cnf.clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_d4v2(cnf_file):
    result = subprocess.run(
        [D4V2_PATH, "-i", cnf_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=3600
    )

    output = result.stdout
    count = None
    elapsed = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("s "):
            try:
                count = int(line.split()[1])
            except (IndexError, ValueError):
                pass
        if "Elapsed time:" in line:
            try:
                elapsed = float(line.split("Elapsed time:")[1].split()[0])
            except (IndexError, ValueError):
                pass

    return count, elapsed

def main():
    density = int(os.environ["SLURM_ARRAY_TASK_ID"])

    random.seed(1234 + density)
    np.random.seed(1234 + density)

    counts = []
    times = []


    for i in range(RUNS):

        bitmap = random_bitmap(m, n, density)

        cnf, vpool = encode_tomography(
            ENCODING, bitmap, m, n, rowcol=True, diags=True
        )

        cnf_file = f"tmp_d4v2_{density}_{i}.cnf"
        write_cnf_for_d4v2(cnf, cnf_file, m, n)

        try:
            count, elapsed = run_d4v2(cnf_file)
        except subprocess.TimeoutExpired:
            count, elapsed = None, None
            print(f"d4v2 timed out (density={density}, i={i})")
        finally:
            os.remove(cnf_file)

        if count is not None and elapsed is not None:
            counts.append(count)
            times.append(elapsed)
            print(f"density={density}, i={i}, count={count}, time={elapsed}", flush=True)
    
    with open(f"d4v2_results_{density}.txt", "w") as f:
        f.write(f"density={density}\n")
        f.write(f"median_count={statistics.median(counts)}\n")
        f.write(f"median_time={statistics.median(times)}\n")

if __name__ == "__main__":
    main()