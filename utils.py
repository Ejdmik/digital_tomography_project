from pysat.card import *
from pysat.solvers import *
import numpy as np
import math
import statistics
import csv

def get_r_vector(bitmap):
    return np.sum(bitmap, axis=1)

def get_c_vector(bitmap):
    return np.sum(bitmap, axis=0)

def get_a_vector(bitmap):
    flipped = np.fliplr(bitmap)
    antidiag_sums = [int(np.sum(np.diag(flipped, k=k))) for k in range(-bitmap.shape[0] + 1, bitmap.shape[1])]
    return np.flip(np.array(antidiag_sums))

def get_b_vector(bitmap):
    diag_sums = [np.sum(np.diag(bitmap, k=k)) for k in range(-bitmap.shape[0]+1, bitmap.shape[1])]
    return np.flip(diag_sums)

# slope diagonals image encodings

"""
Diagonals with slope 1/k
. . . . x x
. . x x . .
x x . . . . (when k = 2)
"""

def get_a_with_slope_vector(bitmap, k):
    m, n = bitmap.shape
    a_with_slope_vector = []
    total_groups = m + int(np.ceil(n / k)) - 1
    
    for i in range(total_groups):
        diag_sum = 0
        for j in range(max(0, i - m + 1), min(i + 1, int(np.ceil(n / k)))):
            for l in range(k):
                col = j * k + l
                row = i - j
                if col < n and row < m:
                    diag_sum += bitmap[row, col]
        a_with_slope_vector.append(diag_sum)
    
    return np.array(a_with_slope_vector)

def get_b_with_slope_vector(bitmap, k):
    m, n = bitmap.shape
    b_with_slope_vector = []
    total_groups = m + int(np.ceil(n / k)) - 1

    for i in range(total_groups):
        diag_sum = 0
        for j in range(max(0, i - m + 1), min(i + 1, int(np.ceil(n / k)))):
            for l in range(k):
                col = n - 1 - (j * k + l)
                row = i - j
                if 0 <= col < n and 0 <= row < m:
                    diag_sum += bitmap[row, col]
        b_with_slope_vector.append(diag_sum)

    return np.array(b_with_slope_vector)

# frames image encoding

"""
xxxxxxxxxx
x********x
x*@@@@@@*x
x*@....@*x
x*@.$$.@*x
x*@....@*x
x*@@@@@@*x
x********x
xxxxxxxxxx
"""

def get_frames_vector(bitmap):
    m, n = bitmap.shape
    frames_num = min(math.ceil(m/2), math.ceil(n/2))
    frames_vector = []
    
    for i in range(frames_num):
        frame_sum = 0
        if (i == frames_num - 1) and min(m,n) % 2 == 1:
            # special cases
            if m>n:
                frame_sum += np.sum(bitmap[i:m-i, i])
            elif n>m:
                frame_sum += np.sum(bitmap[i, i:n-i])
            else:
                frame_sum += bitmap[i,i]
        else:
            frame_sum += np.sum(bitmap[i,i:n-i])            # top row
            frame_sum += np.sum(bitmap[m-i-1,i:n-i])        # bottom row
            frame_sum += np.sum(bitmap[i+1:m-i-1, i])       # left edge without corner
            frame_sum += np.sum(bitmap[i+1:m-i-1, n-i-1])   # right edge without corner

        frames_vector.append(frame_sum)
    
    return np.array(frames_vector)

#-------------------------------------------------------------------------------------------------

def encode_rows(cnf, vpool, rows, encoding, n):
    for i, row_i in enumerate(rows, start=1):
        lits = [vpool.id((i, j)) for j in range(1, n+1)]
        cnf.extend(CardEnc.equals(lits=lits, bound=row_i, vpool=vpool, encoding=encoding))

def encode_cols(cnf, vpool, cols, encoding, m):
    for j, col_j in enumerate(cols, start=1):
        lits = [vpool.id((i, j)) for i in range(1, m+1)]
        cnf.extend(CardEnc.equals(lits=lits, bound=col_j, vpool=vpool, encoding=encoding))

def encode_a_diagonals(cnf, vpool, a, encoding, m, n):
    for k, a_k in enumerate(a, start=1):
        lits = []

        current_point = (k, 1) if k <= m else (m, k - m + 1)

        while current_point[0] >= 1 and current_point[1] <= n:
            lits.append(vpool.id(current_point))
            current_point = (current_point[0] - 1, current_point[1] + 1)

        #print([vpool.obj(l) for l in lits])
        
        cnf.extend(CardEnc.equals(lits=lits, bound=a_k, vpool=vpool, encoding=encoding))

def encode_b_diagonals(cnf, vpool, b, encoding, m, n):
    for l, b_k in enumerate(b, start=1):
        lits = []

        current_point = (l, n) if l <= m else (m, n - (l - m))

        while current_point[0] >= 1 and current_point[1] >= 1:
            lits.append(vpool.id(current_point))
            current_point = (current_point[0] - 1, current_point[1] - 1)

        #print([vpool.obj(l) for l in lits])

        cnf.extend(CardEnc.equals(lits=lits, bound=b_k, vpool=vpool, encoding=encoding))

def encode_a_slope_diagonals(cnf, vpool, a_slope, k, encoding, m, n):
    for p, a_p in enumerate(a_slope, start=1):
        lits = []

        current_point = (p,1) if p <= m else (m, (p-m)*k + 1)

        while current_point[0] >= 1 and current_point[1] <= n:
            lits.append(vpool.id(current_point))
            current_point = (current_point[0] - 1, current_point[1] + 1) if current_point[1] % k == 0 else (current_point[0], current_point[1] + 1)

        #print([vpool.obj(l) for l in lits])    
    
        cnf.extend(CardEnc.equals(lits=lits, bound=a_p, vpool=vpool, encoding=encoding))

def encode_b_slope_diagonals(cnf, vpool, b_slope, k, encoding, m, n):
    for l, b_p in enumerate(b_slope, start=1):
        lits = []

        current_point = (l,n) if l <= m else (m, n - (l-m)*k)

        switch = k - 1

        while current_point[0] >= 1 and current_point[1] >= 1:
            lits.append(vpool.id(current_point))
            current_point = (current_point[0] - 1, current_point[1] - 1) if switch == 0 else (current_point[0], current_point[1] - 1)

            if switch == 0:
                switch = k - 1
            else:
                switch -= 1

        #print([vpool.obj(l) for l in lits])
        
        cnf.extend(CardEnc.equals(lits=lits, bound=b_p, vpool=vpool, encoding=encoding))

def encode_frames(cnf, vpool, frames, encoding, m, n):
    for k, f_k in enumerate(frames, start=0):
        lits = []

        top = k
        bottom = m - k - 1
        left = k
        right = n - k - 1

        # top edge
        for j in range(left, right + 1):
            lits.append(vpool.id((top + 1, j + 1)))  # +1 if vpool uses 1-based indexing

        # bottom edge
        for j in range(left, right + 1):
            lits.append(vpool.id((bottom + 1, j + 1)))

        # left edge
        for i in range(top, bottom + 1):
            lits.append(vpool.id((i + 1, left + 1)))

        # right edge
        for i in range(top, bottom + 1):
            lits.append(vpool.id((i + 1, right + 1)))

        # remove duplicates
        lits = list(dict.fromkeys(lits))

        # print([vpool.obj(l) for l in lits])
        
        cnf.extend(CardEnc.equals(lits=lits, bound=f_k, vpool=vpool, encoding=encoding))

#------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def show_model(model, vpool, m, n, position=None):
    # Matrix of nans
    matrix = np.empty((m, n), dtype=int)
    matrix[:] = -1

    for var in model:
        if var > 0:
            name = vpool.obj(var)
            if isinstance(name, tuple) and len(name) == 2:
                matrix[name[0] - 1, name[1] - 1] = 0
        else:
            assert var < 0
            name = vpool.obj(-var)
            if isinstance(name, tuple) and len(name) == 2:
                matrix[name[0] - 1, name[1] - 1] = 1

    assert np.max(matrix) == 1 and np.min(matrix) == 0

    matrix = matrix.astype(bool)

    plt.clf()  # Clear the current figure
    plt.imshow(matrix, cmap='gray', interpolation='nearest')

    if position is not None:
        i, j = position
        plt.scatter(j - 1, i - 1, color='red')  # Highlight the current position

    plt.xticks(ticks=np.arange(n), labels=np.arange(1, n + 1))
    plt.yticks(ticks=np.arange(m), labels=np.arange(1, m + 1))
    plt.grid(False)
    plt.show()


def model_to_bitmap(model,vpool):
    positive_cells = [vpool.obj(l) for l in model if l > 0 and vpool.obj(abs(l)) is not None]

    if not positive_cells:
        return np.zeros((0, 0))

    max_row = max(r for r, c in positive_cells)
    max_col = max(c for r, c in positive_cells)

    bitmap = np.zeros((max_row, max_col), dtype=int)
    for row, col in positive_cells:
        bitmap[row - 1, col - 1] = 1

    return bitmap

def show_bitmap(bitmap, position=None):
    matrix = bitmap.astype(bool)

    h, w = matrix.shape

    plt.clf()
    plt.imshow(~matrix, cmap='gray', interpolation='nearest')

    if position is not None:
        i, j = position
        plt.scatter(j - 1, i - 1, color='red')

    plt.xticks(ticks=np.arange(w), labels=np.arange(1, w + 1))
    plt.yticks(ticks=np.arange(h), labels=np.arange(1, h + 1))
    plt.grid(False)
    plt.show()


def bitmap_to_model(bitmap, vpool):
    height, width = bitmap.shape
    model = []

    for i in range(height):
        for j in range(width):
            var = vpool.id((i + 1, j + 1))
            if bitmap[i, j] == 1:
                model.append(var)
            else:
                model.append(-var)

    return model

#----------------------------------------------------------------------------
import time
import statistics

# bool rowcol, bool diags, int k_slope (if 0 then it is turned off), frame
def encode_tomography(encoding, bitmap, m, n, rowcol=True, diags=True, k_slope=0, frames=False):
    vpool = IDPool()
    cnf = CNF()

    for i in range(1, m+1):
        for j in range(1, n+1):
            vpool.id((i, j))

    if rowcol:
        rows = get_r_vector(bitmap)
        cols = get_c_vector(bitmap)
        encode_rows(cnf, vpool, rows, encoding, n)
        encode_cols(cnf, vpool, cols, encoding, m)

    if diags:
        a = get_a_vector(bitmap)
        b = get_b_vector(bitmap)
        encode_a_diagonals(cnf, vpool, a, encoding, m, n)
        encode_b_diagonals(cnf, vpool, b, encoding, m, n)

    if k_slope > 0:
        a_slope = get_a_with_slope_vector(bitmap, k_slope)
        b_slope = get_b_with_slope_vector(bitmap, k_slope)
        encode_a_slope_diagonals(cnf, vpool, a_slope, k_slope, encoding, m, n)
        encode_b_slope_diagonals(cnf, vpool, b_slope, k_slope, encoding, m, n)
    
    if frames:
        frames_vec = get_frames_vector(bitmap)
        encode_frames(cnf, vpool, frames_vec, encoding, m, n)
    
    return cnf, vpool

def measure_encoding_performance(encoding, bitmap, m, n, rowcol=True, diags=True, k_slope=0, frames=False):
    """Build and solve the CNF, measuring size and time."""

    t0 = time.time()
    cnf, vpool = encode_tomography(encoding, bitmap, m, n, rowcol, diags, k_slope, frames)
    build_time = time.time() - t0

    t1 = time.time()
    #solver = Cadical195(bootstrap_with=cnf)
    solver = Glucose3(bootstrap_with=cnf)
    sat = solver.solve()
    solve_time = time.time() - t1

    n_vars = max(abs(lit) for clause in cnf.clauses for lit in clause)
    n_clauses = len(cnf.clauses)

    solver.delete()
    del solver

    return {
        "encoding": encoding,
        "vars": n_vars,
        "clauses": n_clauses,
        "build_time": build_time,
        "solve_time": solve_time,
        "satisfiable": sat
    }


# script for RCI cluster which runs each scenario multiple times and computes the average
def run_experiment(bitmap, reps_num=1, rowcol=True, diags=False, k_slope=0, frames=False, scenario_name="unknown", job_id="local"):
    m, n = bitmap.shape
    encodings = [1, 2, 3]  # 1=seqcounter, 2=totalizer, 3=cardnetw

    print(f"{'Scenario':<30} {'Encoding':<10} {'Vars':>8} {'Clauses':>10} "
      f"{'BuildAvg':>10} {'SolveAvg':>10} {'SAT?':>6}")

    filename = f"results_{job_id}.csv"

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        for enc in encodings:
            build_times, solve_times = [], []
            satisfiable = None

            for _ in range(reps_num):
                result = measure_encoding_performance(
                    enc,
                    bitmap,
                    m,
                    n,
                    rowcol=rowcol,
                    diags=diags,
                    k_slope=k_slope,
                    frames=frames
                )

                build_times.append(result["build_time"])
                solve_times.append(result["solve_time"])
                satisfiable = result["satisfiable"]

            build_avg = statistics.mean(build_times)
            solve_avg = statistics.mean(solve_times)

            print(f"{scenario_name:<30} {enc:<10} "
                  f"{result['vars']:>8} {result['clauses']:>10} "
                  f"{build_avg:>10.4f} "
                  f"{solve_avg:>10.4f} {str(satisfiable):>6}")

            # 👇 save structured result
            writer.writerow([
                scenario_name,
                enc,
                result["vars"],
                result["clauses"],
                build_avg,
                solve_avg,
                satisfiable
            ])

    print(f"Saved results to {filename}")




#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# SAT in 3D

# bitmap3d.shape == (m, n, p)

def get_axis_vectors(bitmap, axis):
    return np.sum(bitmap, axis=axis)

from collections import defaultdict

def compute_plane_diagonals(bitmap3d, coords2d, vpool):
    """
    Computes diagonal and antidiagonal SAT literals and bounds
    directly from the bitmap.

    Returns:
        diag_lits, diag_bounds
        anti_lits, anti_bounds
    """

    diag_lits = defaultdict(list)
    diag_bounds = defaultdict(int)

    anti_lits = defaultdict(list)
    anti_bounds = defaultdict(int)

    for i, row in enumerate(coords2d):
        for j, (z,y,x) in enumerate(row):

            lit = vpool.id((z,y,x))
            val = bitmap3d[z-1, y-1, x-1]

            d = i - j
            a = i + j

            diag_lits[d].append(lit)
            diag_bounds[d] += val

            anti_lits[a].append(lit)
            anti_bounds[a] += val

    diag_keys = sorted(diag_lits.keys())
    anti_keys = sorted(anti_lits.keys())

    diag_lits_list = [diag_lits[k] for k in diag_keys]
    diag_bounds_list = [diag_bounds[k] for k in diag_keys]

    anti_lits_list = [anti_lits[k] for k in anti_keys]
    anti_bounds_list = [anti_bounds[k] for k in anti_keys]

    return diag_lits_list, diag_bounds_list, anti_lits_list, anti_bounds_list


def get_boxes_vector(bitmap3d):
    m,n,p = bitmap3d.shape
    boxes_num = min((m+1)//2, (n+1)//2, (p+1)//2)

    boxes = np.zeros(boxes_num, dtype=int)

    for z in range(m):
        for y in range(n):
            for x in range(p):

                dist = min(z,m-1-z,y,n-1-y,x,p-1-x)
                boxes[dist] += bitmap3d[z,y,x]

    return boxes


#--------------------------------------------------------------------------------------------------------------

def encode_xy(cnf, vpool, P_xy, encoding, p):
    m, n = P_xy.shape
    
    for i in range(m):
        for j in range(n):
            lits = [
                vpool.id((i+1, j+1, k+1))
                for k in range(p)
            ]
            cnf.extend(
                CardEnc.equals(
                    lits=lits,
                    bound=P_xy[i, j],
                    vpool=vpool,
                    encoding=encoding
                )
            )

def encode_xz(cnf, vpool, P_xz, encoding, n):
    m, p = P_xz.shape
    
    for i in range(m):
        for k in range(p):
            lits = [
                vpool.id((i+1, j+1, k+1))
                for j in range(n)
            ]
            cnf.extend(
                CardEnc.equals(
                    lits=lits,
                    bound=P_xz[i, k],
                    vpool=vpool,
                    encoding=encoding
                )
            )

def encode_yz(cnf, vpool, P_yz, encoding, m):
    n, p = P_yz.shape
    
    for j in range(n):
        for k in range(p):
            lits = [
                vpool.id((i+1, j+1, k+1))
                for i in range(m)
            ]
            cnf.extend(
                CardEnc.equals(
                    lits=lits,
                    bound=P_yz[j, k],
                    vpool=vpool,
                    encoding=encoding
                )
            )


def encode_plane_diagonals_from_bitmap(cnf, vpool, bitmap3d, coords2d, encoding):

    diag_lits, diag_bounds, anti_lits, anti_bounds = \
        compute_plane_diagonals(bitmap3d, coords2d, vpool)

    for lits, bound in zip(diag_lits, diag_bounds):

        if bound > len(lits):
            raise ValueError(f"Impossible diagonal bound {bound}>{len(lits)}")

        cnf.extend(CardEnc.equals(
            lits=lits,
            bound=bound,
            vpool=vpool,
            encoding=encoding
        ))

    for lits, bound in zip(anti_lits, anti_bounds):

        if bound > len(lits):
            raise ValueError(f"Impossible anti bound {bound}>{len(lits)}")

        cnf.extend(CardEnc.equals(
            lits=lits,
            bound=bound,
            vpool=vpool,
            encoding=encoding
        ))

def plane_xy_coords(m,n,p,z):
    return [[(z,y,x) for x in range(1,p+1)] for y in range(1,n+1)]

def plane_xz_coords(m,n,p,y):
    return [[(z,y,x) for x in range(1,p+1)] for z in range(1,m+1)]

def plane_yz_coords(m,n,p,x):
    return [[(z,y,x) for y in range(1,n+1)] for z in range(1,m+1)]

def encode_all_diagonals(cnf, vpool, bitmap3d, encoding):

    m,n,p = bitmap3d.shape

    # XY planes
    for z in range(1,m+1):
        coords = plane_xy_coords(m,n,p,z)
        encode_plane_diagonals_from_bitmap(
            cnf, vpool, bitmap3d, coords, encoding
        )

    # XZ planes
    for y in range(1,n+1):
        coords = plane_xz_coords(m,n,p,y)
        encode_plane_diagonals_from_bitmap(
            cnf, vpool, bitmap3d, coords, encoding
        )

    # YZ planes
    for x in range(1,p+1):
        coords = plane_yz_coords(m,n,p,x)
        encode_plane_diagonals_from_bitmap(
            cnf, vpool, bitmap3d, coords, encoding
        )


def encode_boxes(cnf, vpool, boxes, encoding, m, n, p):
    for k, b_k in enumerate(boxes, start=0):
        lits = []

        front  = k
        back   = m - k - 1
        top    = k
        bottom = n - k - 1
        left   = k
        right  = p - k - 1

        # --- Z faces (front & back) ---
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                lits.append(vpool.id((front + 1, y + 1, x + 1)))
                lits.append(vpool.id((back  + 1, y + 1, x + 1)))

        # --- Y faces (top & bottom) ---
        for z in range(front, back + 1):
            for x in range(left, right + 1):
                lits.append(vpool.id((z + 1, top    + 1, x + 1)))
                lits.append(vpool.id((z + 1, bottom + 1, x + 1)))

        # --- X faces (left & right) ---
        for z in range(front, back + 1):
            for y in range(top, bottom + 1):
                lits.append(vpool.id((z + 1, y + 1, left  + 1)))
                lits.append(vpool.id((z + 1, y + 1, right + 1)))

        # remove duplicates (edges & corners appear multiple times)
        lits = list(dict.fromkeys(lits))

        cnf.extend(CardEnc.equals(
            lits=lits,
            bound=b_k,
            vpool=vpool,
            encoding=encoding
        ))


#--------------------------------------------------------------------------------------------------------------------

def encode_tomography3d(encoding, bitmap3d, rowcol3d=True, diags3d=True, boxes=False):
    m, n, p = bitmap3d.shape

    vpool = IDPool()
    cnf = CNF()

    for z in range(1, m+1):
        for y in range(1, n+1):
            for x in range(1, p+1):
                vpool.id((z, y, x))

    if rowcol3d:
        P_xy = get_axis_vectors(bitmap3d, axis=2)
        P_xz = get_axis_vectors(bitmap3d, axis=1)
        P_yz = get_axis_vectors(bitmap3d, axis=0)

        encode_xy(cnf, vpool, P_xy, encoding, p)
        encode_xz(cnf, vpool, P_xz, encoding, n)
        encode_yz(cnf, vpool, P_yz, encoding, m)

    
    if diags3d:
        encode_all_diagonals(cnf, vpool, bitmap3d, encoding)


    if boxes:
        boxes_vec = get_boxes_vector(bitmap3d)
        encode_boxes(cnf, vpool, boxes_vec, encoding, m, n, p)

    return cnf, vpool


def measure_encoding_performance3d(encoding, bitmap3d, rowcol3d=True, diags3d=True, boxes=False):

    t0 = time.time()
    cnf, vpool = encode_tomography3d(
        encoding,
        bitmap3d,
        rowcol3d=rowcol3d,
        diags3d=diags3d,
        boxes=boxes
    )
    build_time = time.time() - t0

    t1 = time.time()
    #solver = Cadical195(bootstrap_with=cnf)
    solver = Glucose3(bootstrap_with=cnf)
    sat = solver.solve()
    solve_time = time.time() - t1

    n_vars = max(abs(lit) for clause in cnf.clauses for lit in clause)
    n_clauses = len(cnf.clauses)

    return {
        "encoding": encoding,
        "vars": n_vars,
        "clauses": n_clauses,
        "build_time": build_time,
        "solve_time": solve_time,
        "satisfiable": sat
    }


def run_experiment3d(bitmap3d, reps_num=3, rowcol3d=True, diags3d=False, boxes=False, scenario_name="unknown", job_id="local"):

    encodings = [1, 2, 3]  # seqcounter, totalizer, cardnetw

    print(f"{'Scenario':<30} {'Encoding':<10} {'Vars':>10} {'Clauses':>12} "
          f"{'BuildAvg':>10} {'SolveAvg':>10} {'SAT?':>6}")

    filename = f"results3d_{job_id}.csv"

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        for enc in encodings:

            build_times = []
            solve_times = []
            satisfiable = None

            for _ in range(reps_num):
                result = measure_encoding_performance3d(
                    enc,
                    bitmap3d,
                    rowcol3d=rowcol3d,
                    diags3d=diags3d,
                    boxes=boxes
                )

                build_times.append(result["build_time"])
                solve_times.append(result["solve_time"])
                satisfiable = result["satisfiable"]

            build_avg = statistics.mean(build_times)
            solve_avg = statistics.mean(solve_times)

            # Print aligned row
            print(f"{scenario_name:<30} {enc:<10} "
                  f"{result['vars']:>10} {result['clauses']:>12} "
                  f"{build_avg:>10.4f} {solve_avg:>10.4f} {str(satisfiable):>6}")

            # Save structured results
            writer.writerow([
                scenario_name,
                enc,
                result['vars'],
                result['clauses'],
                build_avg,
                solve_avg,
                satisfiable
            ])

    print(f"Saved results to {filename}")


#-------------------------------------------------------------------
import random

def random_vector(n, total, max_val):
    x = []
    remaining_sum = total

    for i in range(n):
        remaining_slots = n - i

        if remaining_slots == 1:
            # last element must take what's left
            x.append(remaining_sum)
        else:
            low = max(0, remaining_sum - max_val * (remaining_slots - 1))
            high = min(max_val, remaining_sum)

            val = random.randint(low, high)
            x.append(val)
            remaining_sum -= val

    return x


def random_vector2(n, total, max_val):
    x = [0] * n
    available = list(range(n))  

    for _ in range(total):
        i = random.choice(available)
        x[i] += 1

        if x[i] == max_val:
            available.remove(i)

    return x

import numpy as np

def random_vector3(n, total, max_val):
    if total < n:
        x = [0] * n
        for _ in range(total):
            x[random.randrange(n)] += 1
        return x

    if total > n * max_val // 2:
        comp_total = n * max_val - total
        comp = random_vector3(n, comp_total, max_val)
        return [max_val - x for x in comp]

    while True:
        cuts = sorted(random.sample(range(total + 1), n - 1))
        cuts = [0] + cuts + [total]
        x = [cuts[i+1] - cuts[i] for i in range(n)]
        if all(v <= max_val for v in x):
            return x


def random_diag_vector(m, n, total):
    length = m + n - 1

    max_vals = []
    for i in range(length):
        upper = min(i + 1, m, n, m + n - i - 1)
        max_vals.append(upper)

    vec = []
    remaining_sum = total

    for i in range(length):
        remaining_slots = length - i

        if remaining_slots == 1:
            val = remaining_sum
        else:
            remaining_max = sum(max_vals[i+1:])

            low = max(0, remaining_sum - remaining_max)
            high = min(max_vals[i], remaining_sum)

            val = random.randint(low, high)

        vec.append(val)
        remaining_sum -= val

    return np.array(vec)



def random_diag_vector2(m, n, total):
    length = m + n - 1

    max_vals = []
    for i in range(length):
        upper = min(i + 1, m, n, m + n - i - 1)
        max_vals.append(upper)


    x = [0] * length
    available = list(range(length))

    for _ in range(total):
        i = random.choice(available)
        x[i] += 1

        if x[i] == max_vals[i]:
            available.remove(i)

    return np.array(x)


def random_bitmap(m, n, density):
    total_cells = m * n

    if density > total_cells:
        raise ValueError("Density exceeds number of cells")

    grid = [0] * total_cells

    ones_positions = random.sample(range(total_cells), density)
    for pos in ones_positions:
        grid[pos] = 1

    return np.array(grid).reshape((m, n))

def random_bitmap3d(m, n, p, density):
    total_cells = m * n * p

    if density > total_cells:
        raise ValueError("Density exceeds number of cells")

    grid = [0] * total_cells

    ones_positions = random.sample(range(total_cells), density)
    for pos in ones_positions:
        grid[pos] = 1

    return np.array(grid).reshape((m, n, p))


def random_axis_vector(m, n, total, max_val):
    size = m * n
    x = []
    remaining_sum = total

    for i in range(size):
        remaining_slots = size - i

        if remaining_slots == 1:
            x.append(remaining_sum)
        else:
            low = max(0, remaining_sum - max_val * (remaining_slots - 1))
            high = min(max_val, remaining_sum)

            val = random.randint(low, high)
            x.append(val)
            remaining_sum -= val

    return np.array(x).reshape((m, n))


def random_axis_vector2(m, n, total, max_val):
    size = m * n

    x = [0] * size
    available = list(range(size))

    for _ in range(total):
        i = random.choice(available)
        x[i] += 1

        if x[i] == max_val:
            available.remove(i)

    return np.array(x).reshape((m, n))