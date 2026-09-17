#!/usr/bin/env python3
"""Fig: PIC/MRR transient temperature, from the ANSYS transient thermal
simulation (OIO3D GPU/EIC/aPIC stack, Coenen et al. TCPMT 2026 material/HTC/
bond-resistance parameters, FLOP-based GPU power v2).

MRR-location (aPIC-center) temperature vs time, both models. This raw T(t)
trace is the input to the optical-domain stall derivation in
plot_stall_physical.py / fig_stall_derivation.png -- that figure, not this
one, now carries the tuning-delay derivation.
"""
import re
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#008300"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"


def parse_prvar(path):
    times, temps = [], []
    with open(path) as f:
        started = False
        for line in f:
            line = line.strip()
            if line.startswith("TIME"):
                started = True
                continue
            if started:
                m = re.match(r"^([\d.Ee+-]+)\s+([\d.Ee+-]+)$", line)
                if m:
                    times.append(float(m.group(1)))
                    temps.append(float(m.group(2)))
    return times, temps


datasets = [
    ("Mixtral-8x7B", "mixtral8x7b_pic_transient_L4seq4096.csv", BLUE),
    ("Qwen-MoE-14.3B", "qwenmoe_pic_transient_L4seq4096.csv", ORANGE),
    ("LLaMA-MoE-6.7B", "llamamoe_pic_transient_L4seq4096.csv", GREEN),
]

fig, ax1 = plt.subplots(1, 1, figsize=(4.2, 2.9))
fig.patch.set_facecolor("#fcfcfb")

for name, path, color in datasets:
    t, T = parse_prvar(path)
    ax1.plot(t, T, color=color, linewidth=1.6, label=name)

ax1.set_facecolor("#fcfcfb")
ax1.grid(True, color=GRID, linewidth=0.7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_color("#c3c2b7")
ax1.spines["bottom"].set_color("#c3c2b7")
ax1.tick_params(colors=SEC_INK, labelsize=9)
ax1.xaxis.label.set_color(INK)
ax1.yaxis.label.set_color(INK)

ax1.set_ylabel("aPIC / MRR temperature (°C)", fontsize=10)
ax1.set_xlabel("Time (s)", fontsize=10)
ax1.legend(frameon=True, fontsize=8.5, loc="lower right", labelcolor=INK,
           facecolor="#fcfcfb", edgecolor="none", framealpha=0.92)

fig.tight_layout()
fig.savefig("fig_pic_mrr_transient.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
fig.savefig("fig_pic_mrr_transient.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("wrote fig_pic_mrr_transient.png / .pdf")
