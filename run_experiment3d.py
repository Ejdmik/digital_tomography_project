import numpy as np
import sys
import os
from utils import run_experiment3d
from scipy.ndimage import binary_erosion, binary_dilation

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


def bitmap_to_3d(cat_bitmap, depth=25):
    H, W = cat_bitmap.shape
    D = depth

    volume = np.zeros((D, H, W), dtype=int)

    # Normalize depth to [-1, 1]
    z_vals = np.linspace(-1, 1, D)

    for i, z in enumerate(z_vals):
        layer = cat_bitmap.copy()

        # Create "rounding" effect:
        # middle layers = full size
        # outer layers = slightly eroded
        shrink_factor = abs(z)

        if shrink_factor > 0.6:
            # strongly shrink outer layers
            layer = binary_erosion(layer, iterations=2)
        elif shrink_factor > 0.3:
            layer = binary_erosion(layer, iterations=1)
        elif shrink_factor < 0.1:
            # slight bulge in the middle
            layer = binary_dilation(layer, iterations=1)

        volume[i] = layer.astype(int)

    return volume


cat_bitmap_3d = bitmap_to_3d(cat_bitmap, depth=25)

SCENARIOS = [
        {"name": "axes only", "rowcol3d": True, "diags3d": False, "boxes": False},
        {"name": "diags only", "rowcol3d": False, "diags3d": True, "boxes": False},
        {"name": "boxes only", "rowcol3d": False, "diags3d": False, "boxes": True},
        {"name": "axes + diags", "rowcol3d": True, "diags3d": True, "boxes": False},
        {"name": "axes + boxes", "rowcol3d": True, "diags3d": False, "boxes": True},
        {"name": "axes + diags + boxes", "rowcol3d": True, "diags3d": True, "boxes": True},
    ]
if __name__ == "__main__":
    idx = int(sys.argv[1])
    scenario = SCENARIOS[idx]

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

    print(f"Running scenario {idx}: {scenario['name']}")

    run_experiment3d(
        cat_bitmap_3d,
        reps_num=1,
        rowcol3d=scenario["rowcol3d"],
        diags3d=scenario["diags3d"],
        boxes=scenario["boxes"],
        scenario_name=scenario["name"],
        job_id=job_id
    )

    print("Finished.")