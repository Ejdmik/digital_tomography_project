# import matplotlib.pyplot as plt

# densities = []
# values = []
# med_times = []

# with open("semi_random_sat_rc_solvability5.txt") as f:
#     for line in f:
#         # random_sat (or exact_sat)
#         line = line.strip()
#         parts = line.replace(" ", "").split(",")

#         d = int(parts[0].split("=")[1])
#         v = int(parts[1].split("=")[1])

#         densities.append(d/225)
#         values.append(v/1000) # v/1000 if SAT ratio

#         # difficulty_sat
#         # line = line.strip()
#         # parts = line.replace(" ", "").split(",")

#         # d = int(parts[0].split("=")[1])
#         # med = float(parts[1].split("=")[1])

#         # densities.append(d/216)
#         # med_times.append(med)


# data = sorted(zip(densities, values))
# densities, values = zip(*data)

# #data = sorted(zip(densities, med_times))
# #densities, med_times = zip(*data)

# # plot
# plt.figure()

# plt.plot(densities, values)

# #plt.plot(densities, med_times)

# #plt.yscale("log")


# plt.xlabel("Density")

# plt.ylabel("SAT ratio")
# #plt.ylabel("Median time (Seconds)")
# #plt.ylabel("Number of solutions")

# plt.title("SAT solvability (15x15 grid)")
# #plt.title("SAT solving time vs Density (15x15 grid)")
# #plt.title("Median number of solutions using solver EXACT")

# plt.grid(True)

# plt.savefig("semi_random_sat_solvability_rc_graph5.png", dpi=300)
# plt.show()


#semi_random_sat_solvability:
#rc: 1 = balls into bins both, 2 = jeden tak, druhy tak, 3 = oba slozitejsi zpusob, 4 = oba slozitejsi zpusob zpermutovany, 5 = random_vector3 oba
#ab: 1 = slozitejsi zpusob oba

#3d semi random
# 1 = balls into bins, 2 = vsechno slozitejsi zpusob, 3 = vsehcno slozite a zpermutovany ... to nemelo ani smysl


import matplotlib.pyplot as plt

data = {}
with open("ganak_results_6x6_rc_seq.txt") as f:
    current = {}
    for line in f:
        line = line.strip()
        if line.startswith("density="):
            if current:
                data[current["density"]] = current
            current = {"density": int(line.split("=")[1])}
        elif line.startswith("median_count="):
            current["median_count"] = float(line.split("=")[1])
        elif line.startswith("median_time="):
            current["median_time"] = float(line.split("=")[1])
    if current:
        data[current["density"]] = current

# Sort by density and normalize
rows = sorted(data.values(), key=lambda x: x["density"])
x = [r["density"] / 36.0 for r in rows]
counts = [r["median_count"] for r in rows]
times = [r["median_time"] for r in rows]

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(x, counts, color="steelblue", marker="o", markersize=3, label="Median count")
ax1.set_xlabel("Density")
ax1.set_ylabel("Median number of solutions", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_yscale("log")

ax2 = ax1.twinx()
ax2.plot(x, times, color="tomato", marker="s", markersize=3, label="Median time")
ax2.set_ylabel("Median solving time (s)", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

plt.title("Ganak: 6×6 bitmap — solutions and solving time by density")
fig.tight_layout()
plt.savefig("ganak_results_6x6_rc_seq.png", dpi=150)
plt.show()