# with open("difficulty_sat_10x10x10_tot_data.txt") as fin, open("difficulty_sat_10x10x10_tot_data2.txt", "w") as fout:
#     for line in fin:
#         parts = line.strip().split(", ")
        
#         density = parts[0]
#         median = parts[2].split("=")[1]
        
#         fout.write(f"{density}, median={median}\n")


def load_data(filename):
    densities = []
    times = []

    with open(filename) as f:
        for line in f:
            parts = line.strip().split(", ")

            d = int(parts[0].split("=")[1])
            t = float(parts[1].split("=")[1])

            densities.append(d/225)
            times.append(t*1000)

    # sort by density
    data = sorted(zip(densities, times))
    densities, times = zip(*data)

    return densities, times

d_seq, t_seq = load_data("difficulty_sat_15x15_rcab_seq_data.txt")
d_tot, t_tot = load_data("difficulty_sat_15x15_rcab_tot_data.txt")
d_card, t_card = load_data("difficulty_sat_15x15_rcab_card_data.txt")

import matplotlib.pyplot as plt

plt.plot(d_seq, t_seq, label="Sequential Counter")
plt.plot(d_tot, t_tot, label="Totalizer")
plt.plot(d_card, t_card, label="Cardinality Network")

plt.xlabel("Density")
plt.ylabel("Median Time (Miliseconds)")
plt.legend()

# optional but recommended (your data spans scales)
#plt.yscale("log")
plt.title("Comparison of encodings")

plt.grid(True)

plt.savefig("difficulty_sat_15x15_rcab_all_graph.png", dpi=300)
plt.show()