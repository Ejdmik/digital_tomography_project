# import matplotlib.pyplot as plt

# densities = []
# values = []
# med_times = []
# raw_densities = []


# with open("semi_random_sat_rcab_solvability.txt") as f:
#     for line in f:
#         # random_sat (or exact_sat)
#         line = line.strip()
#         parts = line.replace(" ", "").split(",")

#         d = int(parts[0].split("=")[1])
#         v = float(parts[1].split("=")[1])

#         raw_densities.append(d)
#         densities.append(d/225)
#         values.append(v/1000) # v/1000 if SAT ratio

#         # difficulty_sat
#         # line = line.strip()
#         # parts = line.replace(" ", "").split(",")

#         # d = int(parts[0].split("=")[1])
#         # med = float(parts[1].split("=")[1])

#         # densities.append(d/225)
#         # med_times.append(med*1000)


# data = sorted(zip(raw_densities, densities, values))
# raw_densities, densities, values = zip(*data)


# # data = sorted(zip(densities, med_times))
# # densities, med_times = zip(*data)

# from math import comb

# def count_vectors(n, m, k):
#     total = 0
#     for j in range(k // (m + 1) + 1):
#         total += (-1)**j * comb(n, j) * comb(n + k - j*(m+1) - 1, k - j*(m+1))
#     return total

# theoretical = []

# for k in raw_densities:
#     denom = count_vectors(15, 15, k)
    
#     if denom == 0:
#         theoretical.append(float('nan'))  # avoid division by zero
#     else:
#         val = comb(225, k) / (denom ** 2)
#         val = val * 10**(min(k, 225-k)/11.5)
#         theoretical.append(val)

# # plot
# plt.figure()
# plt.axvline(0.5, color='grey', linestyle='--', linewidth=0.5)

# # # plt.plot(densities2, counts, label="Experimental (EXACT) - mean", color="deepskyblue")


# plt.plot(densities, values, marker="x")
# #plt.axhline(y=4e31, linestyle='--', label="Uniform average estimate", color="green")
# #plt.plot(densities, theoretical, linestyle='--', label="Density-aware estimate (modified)", color="darkorange")

# #plt.plot(densities, med_times)

# #plt.yscale("log")
# #plt.legend(loc='center', bbox_to_anchor=(0.5, 0.18))



# plt.xlabel("Density")

# plt.ylabel("SAT Ratio")
# #plt.ylabel("Solving Time (ms)")
# #plt.ylabel("Number of Solutions")

# plt.grid(True)

# plt.savefig("random_sat_rcab_graph.png", dpi=300)
# plt.show()

# ----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

def load_file(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: density=121, med_time=0.00037
            parts = line.split(",")
            density = int(parts[0].split("=")[1])
            density = density / 216
            time = float(parts[1].split("=")[1])
            data[density] = time
    return data


seq = load_file("difficulty_sat_6x6x6_seq_data.txt")
tot = load_file("difficulty_sat_6x6x6_tot_new.txt")
card = load_file("difficulty_sat_6x6x6_card_data.txt")

def to_sorted_lists(d):
    xs = sorted(d.keys())
    ys = [d[x] * 1000 for x in xs]  # convert to ms
    return xs, ys

x1, y1 = to_sorted_lists(seq)
x2, y2 = to_sorted_lists(tot)
x3, y3 = to_sorted_lists(card)

plt.figure()

plt.plot(x3, y3, label="CardNetw")
plt.plot(x2, y2, label="Totalizer")
plt.plot(x1, y1, label="SeqCounter")

plt.axvline(0.5, color='grey', linestyle='--', linewidth=0.5)

plt.xlabel("Density")
plt.ylabel("Solving Time (ms)")
plt.legend(loc='center', bbox_to_anchor=(0.5, 0.55))
plt.grid(True)
plt.savefig("difficulty_sat_6x6x6_all_graph.png", dpi=300)

plt.show()