import numpy as np
import sys
import os
from utils import run_experiment

cat_bitmap = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
    [0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0],
    [1,1,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,1,0,0],
    [0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    [0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,1,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    ])

SCENARIOS = [
        {"name": "rowcol only", "rowcol": True, "diags": False, "k_slope": 0, "frames": False},
        {"name": "diags only", "rowcol": False, "diags": True, "k_slope": 0, "frames": False},
        {"name": "slope diags only, k = 2", "rowcol": False, "diags": False, "k_slope": 2, "frames": False},
        {"name": "slope diags only, k = 3", "rowcol": False, "diags": False, "k_slope": 3, "frames": False},
        {"name": "slope diags only, k = 5", "rowcol": False, "diags": False, "k_slope": 5, "frames": False},
        {"name": "slope diags only, k = 10", "rowcol": False, "diags": False, "k_slope": 10, "frames": False},
        {"name": "frames only", "rowcol": False, "diags": False, "k_slope": 0, "frames": True},
        {"name": "rowcol + diag", "rowcol": True, "diags": True, "k_slope": 0, "frames": False},
        {"name": "rowcol + slope2", "rowcol": True, "diags": False, "k_slope": 2, "frames": False},
        {"name": "rowcol + slope3", "rowcol": True, "diags": False, "k_slope": 3, "frames": False},
        {"name": "rowcol + slope5", "rowcol": True, "diags": False, "k_slope": 5, "frames": False},
        {"name": "rowcol + slope10", "rowcol": True, "diags": False, "k_slope": 10, "frames": False},
        {"name": "rowcol + frames", "rowcol": True, "diags": False, "k_slope": 0, "frames": True},
        {"name": "rowcol + diag + slope2", "rowcol": True, "diags": True, "k_slope": 2, "frames": False},
        {"name": "rowcol + diag + slope3", "rowcol": True, "diags": True, "k_slope": 3, "frames": False},
        {"name": "rowcol + diag + slope5", "rowcol": True, "diags": True, "k_slope": 5, "frames": False},
        {"name": "rowcol + diag + slope10", "rowcol": True, "diags": True, "k_slope": 10, "frames": False},
        {"name": "rowcol + diag + frames", "rowcol": True, "diags": True, "k_slope": 0, "frames": True},
        {"name": "rowcol + diag + slope2 + frames", "rowcol": True, "diags": True, "k_slope": 2, "frames": True},
    ]

if __name__ == "__main__":
    idx = int(sys.argv[1])
    scenario = SCENARIOS[idx]

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

    print(f"Running scenario {idx}: {scenario['name']}")

    run_experiment(
        cat_bitmap,
        reps_num=1,
        rowcol=scenario["rowcol"],
        diags=scenario["diags"],
        k_slope=scenario["k_slope"],
        frames=scenario["frames"],
        scenario_name=scenario["name"],
        job_id=job_id
    )

    print("Finished.")