#!/usr/bin/env python3
"""Fig: normalized end-to-end iteration time vs thermal-tuning delay, from the
corrected wafer-scale htsim pipeline (multi-gateway topology fix, corrected
bandwidth, one-time-per-a2a-round delay injection). Replaces combined_logx.png.

Marks the ANSYS-derived (Coenen-grounded transient thermal sim) delay for each
model as a distinct point on its own curve, since that -- not an arbitrary
round-number delay -- is the paper's actual claimed operating point.
"""
import csv
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#008300"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"

sweep = {}
with open("wafer_thermal_sweep_L4seq4096.csv") as f:
    for row in csv.DictReader(f):
        sweep.setdefault(row["model"], []).append(
            (float(row["thermal_delay_ns"]) / 1e6, float(row["final_finish_ms"]))
        )

derived = {}
derived_ms = {}
baselines = {}
with open("wafer_physical_delay_L4seq4096_results.csv") as f:
    for row in csv.DictReader(f):
        if row["derived_from"] == "baseline":
            baselines[row["model"]] = float(row["final_finish_ms"])
        elif row["derived_from"] == "physical_derived":
            derived[row["model"]] = float(row["thermal_delay_ns"]) / 1e6
            derived_ms[row["model"]] = float(row["final_finish_ms"])

datasets = [("mixtral8x7b", "Mixtral-8x7B", BLUE), ("llama-moe-6.7b", "LLaMA-MoE-6.7B", GREEN)]

fig, ax = plt.subplots(figsize=(4.6, 3.4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for key, name, color in datasets:
    pts = sorted(sweep[key])
    baseline = pts[0][1]
    xs = [max(d, 0.5) for d, _ in pts]
    ys = [ms / baseline for _, ms in pts]
    ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4, label=name)

    d_ms = derived[key]
    d_norm_val = derived_ms[key] / baselines[key]
    ax.plot(d_ms, d_norm_val, marker="*", markersize=15, color=color,
             markeredgecolor="white", markeredgewidth=0.6, zorder=6)
    label_offset = (8, 10) if name == "LLaMA-MoE-6.7B" else (10, -14)
    ax.annotate(f"{d_norm_val:.2f}× @ {d_ms:.0f} ms", xy=(d_ms, d_norm_val),
                xytext=label_offset, textcoords="offset points", fontsize=8, color=color,
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.0))

# Qwen-MoE 14.3B: no full sweep (each 512-GPU/EP-64 run took 2-5 hours), so we
# show only its own directly-simulated operating point, not connected by a line.
qwen_base_ms = 710.002038033
qwen_delay_ms = 2697.214061965
qwen_delay_x = 46.76
qwen_norm_val = qwen_delay_ms / qwen_base_ms
ax.plot(qwen_delay_x, qwen_norm_val, marker="*", markersize=15, color=ORANGE,
         markeredgecolor="white", markeredgewidth=0.6, zorder=6,
         linestyle="none", label="Qwen-MoE-14.3B (operating point only)")
ax.annotate(f"{qwen_norm_val:.2f}× @ {qwen_delay_x:.0f} ms", xy=(qwen_delay_x, qwen_norm_val),
            xytext=(-70, 14), textcoords="offset points", fontsize=8, color=ORANGE,
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.0))

ax.set_xscale("log")
ax.set_xlabel("Thermal tuning delay (ms)", fontsize=10)
ax.set_ylabel("Normalized iteration time", fontsize=10)
ax.grid(True, which="both", color=GRID, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=SEC_INK, labelsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)

fig.tight_layout()
fig.savefig("fig_normalized_iteration_time.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
fig.savefig("fig_normalized_iteration_time.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("wrote fig_normalized_iteration_time.png / .pdf")
for key, name, _ in datasets:
    print(f"  {name}: ANSYS-derived delay {derived[key]:.1f} ms")
