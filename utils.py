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

    t0 = time.perf_counter()
    cnf, vpool = encode_tomography(encoding, bitmap, m, n, rowcol, diags, k_slope, frames)
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    #solver = Cadical195(bootstrap_with=cnf)
    solver = Glucose3(bootstrap_with=cnf)
    sat = solver.solve()
    solve_time = time.perf_counter() - t1

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
    encodings = [1, 6, 3]  # 1=seqcounter, 6=totalizer, 3=cardnetwrk

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

def compute_plane_diagonals(bitmap3d, m, n, p):
    result = {
        'xy_a': defaultdict(lambda: {'coords': [], 'bound': 0}),
        'xy_b': defaultdict(lambda: {'coords': [], 'bound': 0}),
        'xz_a': defaultdict(lambda: {'coords': [], 'bound': 0}),
        'xz_b': defaultdict(lambda: {'coords': [], 'bound': 0}),
        'yz_a': defaultdict(lambda: {'coords': [], 'bound': 0}),
        'yz_b': defaultdict(lambda: {'coords': [], 'bound': 0}),
    }

    for i in range(1, m + 1):      # x-axis (rows)
        for j in range(1, n + 1):  # y-axis (cols)
            for k in range(1, p + 1):  # z-axis (depth)

                val = bitmap3d[i-1, j-1, k-1]

                # xy-plane: sum over all k, diagonals in i,j
                result['xy_a'][i + j]['coords'].append((i, j, k))
                result['xy_a'][i + j]['bound'] += val

                result['xy_b'][i - j]['coords'].append((i, j, k))
                result['xy_b'][i - j]['bound'] += val

                # xz-plane: sum over all j, diagonals in i,k
                result['xz_a'][i + k]['coords'].append((i, j, k))
                result['xz_a'][i + k]['bound'] += val

                result['xz_b'][i - k]['coords'].append((i, j, k))
                result['xz_b'][i - k]['bound'] += val

                # yz-plane: sum over all i, diagonals in j,k
                result['yz_a'][j + k]['coords'].append((i, j, k))
                result['yz_a'][j + k]['bound'] += val

                result['yz_b'][j - k]['coords'].append((i, j, k))
                result['yz_b'][j - k]['bound'] += val

    return result


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


def encode_plane_diagonals_from_bitmap(cnf, vpool, bitmap3d, m, n, p, encoding):
    diagonals = compute_plane_diagonals(bitmap3d, m, n, p)

    for family_name, family in diagonals.items():
        for d_idx, data in family.items():
            coords = data['coords']
            bound = data['bound']

            lits = [vpool.id(coord) for coord in coords]

            if bound > len(lits):
                raise ValueError(
                    f"Impossible bound in {family_name}[{d_idx}]: "
                    f"{bound} > {len(lits)}"
                )

            cnf.extend(CardEnc.equals(
                lits=lits,
                bound=bound,
                vpool=vpool,
                encoding=encoding
            ))


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
        encode_plane_diagonals_from_bitmap(cnf, vpool, bitmap3d, encoding)


    if boxes:
        boxes_vec = get_boxes_vector(bitmap3d)
        encode_boxes(cnf, vpool, boxes_vec, encoding, m, n, p)

    return cnf, vpool


def measure_encoding_performance3d(encoding, bitmap3d, rowcol3d=True, diags3d=True, boxes=False):

    t0 = time.perf_counter()
    cnf, vpool = encode_tomography3d(
        encoding,
        bitmap3d,
        rowcol3d=rowcol3d,
        diags3d=diags3d,
        boxes=boxes
    )
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    #solver = Cadical195(bootstrap_with=cnf)
    solver = Glucose3(bootstrap_with=cnf)
    sat = solver.solve()
    solve_time = time.perf_counter() - t1

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

    encodings = [1, 6, 3]  # seqcounter, totalizer, cardnetw

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
    #sequential slot filling
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
    #incremental ball placement
    x = [0] * n
    available = list(range(n))  

    for _ in range(total):
        i = random.choice(available)
        x[i] += 1

        if x[i] == max_val:
            available.remove(i)

    return x

import numpy as np
import random

def random_vector3(n, total, max_val):
    #uniform composition sampling
    if total > n * max_val // 2:
        comp_total = n * max_val - total
        comp = random_vector3(n, comp_total, max_val)
        return [max_val - x for x in comp]
    
    n_prime = n
    M_prime = total + n

    cuts = random.sample(range(1, M_prime), n_prime - 1)
    cuts.sort()

    cuts = [0] + cuts + [M_prime]

    x = [cuts[i+1] - cuts[i] for i in range(n_prime)]

    result = [v - 1 for v in x]

    if any(v > max_val for v in result):
        return random_vector3(n, total, max_val)


    return result


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

def random_diag_vector3(m, n, total):
    length = m + n - 1
    
    max_vals = []
    for i in range(length):
        upper = min(i + 1, m, n, m + n - i - 1)
        max_vals.append(upper)
    
    total_max = sum(max_vals)
    
    while True:
        # Step 1: map to a uniform problem using change of variables
        # sample y_i uniformly from [0, max_vals[i]] with sum = total
        # we do this via rejection: sample unrestricted composition
        # and reject if any component exceeds its bound
        
        # sample a composition of 'total' into 'length' non-negative parts
        # using stars and bars (requires total <= sum of max_vals)
        if total > total_max:
            return None  # infeasible
        
        # use complementation trick if total > total_max / 2
        use_complement = total > total_max // 2
        sample_total = total_max - total if use_complement else total
        
        # stars and bars: place (length - 1) cuts among (sample_total + length - 1) positions
        if sample_total == 0:
            result = [0] * length
            if use_complement:
                result = [max_vals[i] - result[i] for i in range(length)]
            return result
        
        positions = range(1, sample_total + length)
        cuts = sorted(random.sample(positions, length - 1))
        cuts = [0] + cuts + [sample_total + length]
        raw = [cuts[i+1] - cuts[i] - 1 for i in range(length)]
        
        # reject if any component exceeds its bound
        if all(raw[i] <= max_vals[i] for i in range(length)):
            if use_complement:
                return [max_vals[i] - raw[i] for i in range(length)]
            return raw


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





#############################################################################################################



def bitmap_to_latex(arr: np.ndarray) -> None:
    """Print \\def lines for image.tex from a boolean numpy array."""
    arr = np.asarray(arr, dtype=bool)
    rows, cols = arr.shape
    n = rows + cols - 1

    # bitmap: "fx/fy" pairs, 1-indexed, fx=row from top, fy=col from left
    bitmap = ','.join(f'{r+1}/{c+1}' for r, c in np.argwhere(arr))

    rowvec = ','.join(str(int(v)) for v in arr.sum(axis=1))
    colvec = ','.join(str(int(v)) for v in arr.sum(axis=0))

    # "/" anti-diagonal a: cells where row+col == a  (a = 0..n-1)
    adiag = [
        sum(int(arr[r, a - r]) for r in range(rows) if 0 <= a - r < cols)
        for a in range(n)
    ]

    # "\" diagonal b: cells where row-col == b-(cols-1)  (b = 0..n-1)
    bdiag = [
        sum(int(arr[r, r - b + cols - 1]) for r in range(rows) if 0 <= r - b + cols - 1 < cols)
        for b in range(n)
    ]

    print(f'\\def\\rows{{{rows}}}')
    print(f'\\def\\cols{{{cols}}}')
    print(f'\\def\\bitmap{{{bitmap}}}')
    print(f'\\def\\rowvec{{{rowvec}}}')
    print(f'\\def\\colvec{{{colvec}}}')
    print(f'\\def\\adiagvec{{{",".join(str(v) for v in adiag)}}}')
    print(f'\\def\\bdiagvec{{{",".join(str(v) for v in bdiag)}}}')
