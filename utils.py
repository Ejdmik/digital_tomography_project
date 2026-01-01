from pysat.card import *
from pysat.solvers import *
import numpy as np

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
    frames_num = min(1 + m//2, 1 + n//2)
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

        while current_point[0] >= 1 and current_point[1] >= 1:
            lits.append(vpool.id(current_point))
            current_point = (current_point[0] - 1, current_point[1] - 1) if current_point[1] % k == 1 else (current_point[0], current_point[1] - 1)

        #print([vpool.obj(l) for l in lits])
        
        cnf.extend(CardEnc.equals(lits=lits, bound=b_p, vpool=vpool, encoding=encoding))

def encode_frames(cnf, vpool, frames, encoding, m, n):
    for k, f_k in enumerate(frames, start=1):
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

        #print([vpool.obj(l) for l in lits])
        
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
    #plt.grid(True)
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
    solver = Cadical195(bootstrap_with=cnf)
    #solver = Glucose3(bootstrap_with=cnf)
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


# script for RCI cluster which runs each scenario multiple times and computes the average
def run_experiment(bitmap, reps_num=5):
    m, n = bitmap.shape
    scenarios = [
        {"name": "rowcol only", "rowcol": True, "diags": False, "k_slope": 0, "frames": False},
        {"name": "diags only", "rowcol": False, "diags": True, "k_slope": 0, "frames": False},
        {"name": "rowcol + diag", "rowcol": True, "diags": True, "k_slope": 0, "frames": False},
        {"name": "rowcol + slope2", "rowcol": True, "diags": False, "k_slope": 2, "frames": False},
        {"name": "rowcol + slope3", "rowcol": True, "diags": False, "k_slope": 3, "frames": False},
        {"name": "rowcol + frames", "rowcol": True, "diags": False, "k_slope": 0, "frames": True},
        {"name": "rowcol + diag + slope2", "rowcol": True, "diags": True, "k_slope": 2, "frames": False},
        {"name": "rowcol + diag + frames", "rowcol": True, "diags": True, "k_slope": 0, "frames": True},
        # add some more
    ]

    encodings = [1, 2, 3]  # 1=seqcounter, 2=totalizer, 3=cardnetw

    print(f"{'Scenario':<20} {'Encoding':<10} {'Vars':<8} {'Clauses':<10} "
          f"{'BuildAvg':<10} {'SolveAvg':<10} {'SAT?'}")

    for scenario in scenarios:
        for enc in encodings:
            build_times, solve_times = [], []
            satisfiable = None

            for _ in range(reps_num):
                result = measure_encoding_performance(
                    enc,
                    bitmap,
                    m,
                    n,
                    rowcol=scenario["rowcol"],
                    diags=scenario["diags"],
                    k_slope=scenario["k_slope"],
                    frames=scenario["frames"]
                )
                build_times.append(result["build_time"])
                solve_times.append(result["solve_time"])
                satisfiable = result["satisfiable"]

            print(f"{scenario['name']:<20} {enc:<10} "
                  f"{result['vars']:<8} {result['clauses']:<10} "
                  f"{statistics.mean(build_times):<10.4f} "
                  f"{statistics.mean(solve_times):<10.4f} {satisfiable}")
