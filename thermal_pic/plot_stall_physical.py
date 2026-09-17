#!/usr/bin/env python3
"""Fig: optical-domain derivation of the thermal-tuning stall -- replaces the
raw dT/dt-vs-threshold proxy with an actual T(t) -> Delta-lambda(t) ->
controller-residual -> FWHM-penalty-budget chain (see derive_stall_physical.py).

(a) Thermal resonance detuning Delta-lambda_thermal(t) over ten forward passes,
    with the resonance FWHM and DWDM channel spacing marked for scale.
(b) One representative cycle, zoomed: the controller's rate-limited residual
    error eps(t), the acceptable-detuning budget, and the resulting stall
    window where the link is unavailable.
"""
import matplotlib.pyplot as plt
from derive_stall_physical import (
    parse_prvar, simulate_residual, find_stall_windows,
    DLAM_DT_PM_PER_K, R_CTRL_PM_PER_MS, FWHM_PM, EPS_MAX_PM,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#008300"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SEC_INK = "#52514e"
CHANNEL_PM = 1602.0  # 200 GHz DWDM channel spacing @ 1550 nm

datasets = [
    ("Mixtral-8x7B", "mixtral8x7b_pic_transient_L4seq4096.csv", BLUE),
    ("Qwen-MoE-14.3B", "qwenmoe_pic_transient_L4seq4096.csv", ORANGE),
    ("LLaMA-MoE-6.7B", "llamamoe_pic_transient_L4seq4096.csv", GREEN),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.0))
fig.patch.set_facecolor("#fcfcfb")

zoom_data = {}
for name, path, color in datasets:
    t, T = parse_prvar(path)
    eps, dlam = simulate_residual(t, T)
    ax1.plot(t, dlam, color=color, linewidth=1.3, label=name)
    zoom_data[name] = (t, eps, color)

ax1.axhline(-FWHM_PM / 2, color=MUTED, linewidth=0.9, linestyle="--")
ax1.text(0.98, -FWHM_PM / 2, "resonance FWHM/2 ", color=SEC_INK, fontsize=7.5,
          va="bottom", ha="right", transform=ax1.get_yaxis_transform())

# zoomed panel: stall durations are multimodal (they depend on which point in
# the periodic power schedule the thermal peak lands on), so no single window
# duration equals the arithmetic mean used for htsim injection (Section 4.4).
# We therefore show each model's LARGEST steady-state window -- the same
# max value reported in the steady-state range in the text -- skipping the
# first window, which is the one-time startup transient from ambient.
for name, path, color in datasets:
    t, eps, _ = zoom_data[name]
    windows = find_stall_windows(t, eps, EPS_MAX_PM)
    windows = [w for w in windows if w[2] > 1.0][1:]
    w = max(windows, key=lambda w: w[2])
    t0 = w[0] - 0.05
    t1 = w[1] + 0.05
    tw = [(ti - w[0]) * 1000.0 for ti in t if t0 <= ti <= t1]
    ew = [eps[i] for i, ti in enumerate(t) if t0 <= ti <= t1]
    ax2.plot(tw, ew, color=color, linewidth=1.4, label=f"{name} (max {w[2]:.0f} ms)")
    ax2.axvspan(0, w[2], color=color, alpha=0.08)

ax2.axhline(EPS_MAX_PM, color=MUTED, linewidth=1.0, linestyle="--")
ax2.text(0.02, EPS_MAX_PM + 12, "acceptable budget (10% FWHM)",
          color=SEC_INK, fontsize=7.5, va="bottom", ha="left",
          transform=ax2.get_yaxis_transform(),
          bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.0))

for ax in (ax1, ax2):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=SEC_INK, labelsize=8.5)

ax1.set_xlabel("Time (s)", fontsize=9.5)
ax1.set_ylabel("Thermal detuning $\\Delta\\lambda$ (pm)", fontsize=9.5)
ax1.legend(frameon=True, fontsize=7.5, loc="lower right", labelcolor=INK,
            facecolor="#fcfcfb", edgecolor="none", framealpha=0.92)
ax1.set_title("(a) Resonance detuning from workload thermal swing", fontsize=9, color=INK, loc="left")

ax2.set_xlabel("Time (ms), largest steady-state stall window", fontsize=9)
ax2.set_ylabel("Residual detuning $\\varepsilon$ (pm)", fontsize=9.5)
ax2.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK)
ax2.set_title("(b) Controller residual, largest stall window", fontsize=9, color=INK, loc="left")

fig.tight_layout()
fig.savefig("fig_stall_derivation.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
fig.savefig("fig_stall_derivation.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("wrote fig_stall_derivation.png / .pdf")
