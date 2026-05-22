
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

density_counts = defaultdict(list)
density_times = defaultdict(list)


with open("big_ganak3d_xyzdiag_data.txt", "r") as f:
    for line in f:
        line = line.strip()

        # skip irrelevant lines
        if not line.startswith("density="):
            continue

        parts = line.split(",")

        # extract density
        density = int(parts[0].split("=")[1]) / 216

        # extract count
        count_str = parts[1].split("=")[1].strip()

        time = float(parts[2].split("=")[1])

        if count_str != "None":
            density_counts[density].append(float(count_str))
            if time != 0.0:
                density_times[density].append(time*1000)
            else:
                density_times[density].append(5)


densities = sorted(density_counts.keys())
#densities = sorted(density_times.keys())
medians = []
lower = []
upper = []

for d in densities:
    vals = np.log10(density_counts[d])  # work in log space
    #vals = np.log10(density_times[d])
    medians.append(np.median(vals))
    lower.append(np.percentile(vals, 2.5))
    upper.append(np.percentile(vals, 97.5))

medians = np.array(medians)
lower = np.array(lower)
upper = np.array(upper)

fig, ax = plt.subplots()

ax.plot(densities, medians, label="Median")#, color="red")
ax.fill_between(densities, lower, upper, alpha=0.3, label="95% Confidence Interval")#, color="salmon")

ax.set_xlabel("Density")
ax.set_ylabel("Number of Solutions")
#ax.set_ylabel("Solving Time (ms)")
ax.set_yticks(range(0, int(max(medians)) + 2))
ax.set_yticklabels([f"$10^{{{i}}}$" for i in range(0, int(max(medians)) + 2)])
ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.5)

ax.axvline(77/216, color="black", linestyle='--', linewidth=1.5)
ax.axvline(141/216, color="black", linestyle='--', linewidth=1.5)

ax.legend(loc='center', bbox_to_anchor=(0.5, 0.1))
ax.grid(True)
plt.savefig("ganak3d_6x6_xyzdiag.png", dpi=300)
plt.show()