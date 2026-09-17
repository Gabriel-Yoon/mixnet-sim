#!/usr/bin/env python3
"""Fig: routing-aware wavelength reassignment, before vs after, for one wafer
(4x4 GPUs = one LLaMA-MoE 6.7B expert-parallel group, EP=16, TP=1).

Physical model (Section 3 design): each die owns 12 waveguides x 32 lambda = 384 lambda of
egress, 6 waveguides along its row and 6 along its column (192 lambda per direction). A
destination receives on drop rings tuned to the source's wavelengths; reassignment = park
some of destination k's drop rings (~1 nm off-resonance, non-volatile) and un-park spare
rings at destination j for the same wavelengths, while the source EIC re-maps which data
lanes drive which (unchanged) Tx ring modulators. Per-link lambda counts come from the same
allocation rule as htsim_tcp_wafer -lambda-alloc demand (floor 0.1, cap 2x uniform).

Run from thermal_pic/: python3 plot_lambda_reassign.py
"""
import ast
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, ORANGE, MUTED, GRID, INK, SEC_INK, BG = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = ast.literal_eval(open(os.path.join(ROOT, "mixnet-htsim/test/wm_ep16.txt")).read().strip())
N, ROWS, COLS = 16, 4, 4
LAM_PER_DIR, FLOOR, CAP, UNIFORM = 192, 0.1, 2.0, 64
SRC = 0

row = lambda g: g // COLS
col = lambda g: g % COLS


def hops(s, d):
    if s == d:
        return []
    if row(s) == row(d) or col(s) == col(d):
        return [(s, d)]
    mid = row(s) * COLS + col(d)
    return [(s, mid), (mid, d)]


dem = [[(W[g][h] + W[h][g]) if g != h else 0.0 for h in range(N)] for g in range(N)]
load = [[0.0] * N for _ in range(N)]
for s in range(N):
    for d in range(N):
        for u, v in hops(s, d):
            load[u][v] += dem[s][d]


def alloc(u):
    res = {}
    for grp in ([v for v in range(N) if v != u and row(v) == row(u)],
                [v for v in range(N) if v != u and col(v) == col(u)]):
        s = sum(load[u][v] for v in grp)
        fl = FLOOR * s / len(grp)
        w = {v: max(load[u][v], fl) for v in grp}
        ws = sum(w.values())
        lam = {v: LAM_PER_DIR * w[v] / ws for v in grp}
        cap = CAP * LAM_PER_DIR / len(grp)
        for _ in range(4):
            exc = sum(max(0.0, lam[v] - cap) for v in grp)
            for v in grp:
                lam[v] = min(lam[v], cap)
            unc = [v for v in grp if lam[v] < cap]
            if exc <= 0 or not unc:
                break
            tot = sum(lam[v] for v in unc)
            for v in unc:
                lam[v] += exc * lam[v] / tot
        for v in grp:
            res[v] = int(round(lam[v]))
    return res


after = alloc(SRC)
dests = sorted(after)                     # 3 row + 3 col destinations of the source
rx_after = [0] * N
for u in range(N):
    for v, l in alloc(u).items():
        rx_after[v] += l
colsum = [sum(W[i][j] for i in range(N)) for j in range(N)]
loadf = [c / (sum(colsum) / N) for c in colsum]

fig = plt.figure(figsize=(7.2, 5.2))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.35)


def draw_wafer(ax, lam, title):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.7, COLS - 0.3)
    ax.set_ylim(-0.7, ROWS - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    sx, sy = col(SRC), ROWS - 1 - row(SRC)
    # arrows first (behind the boxes); nearer destinations get a small lateral offset so the
    # three row (column) arrows do not lie on top of each other
    for v, l in lam.items():
        x, y = col(v), ROWS - 1 - row(v)
        k = (abs(x - sx) + abs(y - sy))          # 1, 2, 3 hops away
        off = 0.11 * (k - 2)
        if col(v) != sx:
            p0, p1 = (sx, sy + off), (x, y + off)
        else:
            p0, p1 = (sx + off, sy), (x + off, y)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                     shrinkA=10, shrinkB=10, linewidth=0.5 + 3.0 * l / 128,
                                     color=BLUE, alpha=0.75, zorder=1))
    for g in range(N):
        x, y = col(g), ROWS - 1 - row(g)
        hot = loadf[g] >= 1.5
        fc = ORANGE if hot else ("#dcdbd3" if g != SRC else BLUE)
        ax.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64, facecolor=fc, edgecolor="none", zorder=2))
        label = f"G{g}\n{lam[g]}λ" if g in lam else f"G{g}"
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, zorder=3,
                color="white" if (hot or g == SRC) else SEC_INK)


ax0 = fig.add_subplot(gs[0, 0])
draw_wafer(ax0, {v: UNIFORM for v in dests}, "(a) Before: uniform, 64 λ per link")
ax1 = fig.add_subplot(gs[0, 1])
draw_wafer(ax1, after, "(b) After: demand-aware (floor 0.1, cap 2×)")
ax1.text(-0.6, -0.62, "orange = hot-expert GPUs (load ≥ 1.5×); blue = source G0; arrow width ∝ λ",
         fontsize=6.5, color=SEC_INK)

# (c) lambda per destination link of the source, before vs after
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG)
xs = range(len(dests))
ax2.bar([x - 0.19 for x in xs], [UNIFORM] * len(dests), width=0.36, color=MUTED, label="before (uniform)")
ax2.bar([x + 0.19 for x in xs], [after[v] for v in dests], width=0.36, color=BLUE, label="after (demand-aware)")
for x, v in zip(xs, dests):
    ax2.text(x + 0.19, after[v] + 3, str(after[v]), ha="center", fontsize=7, color=INK)
ax2.axhline(CAP * UNIFORM, color=SEC_INK, linewidth=0.8, linestyle="--")
ax2.text(1.5, CAP * UNIFORM + 3, "cap 2× (ring over-provision)", fontsize=6.5, color=SEC_INK, ha="center")
ax2.set_ylim(0, 178)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([f"G{v}\n{'row' if row(v) == row(SRC) else 'col'}" for v in dests], fontsize=7)
ax2.set_xlabel("destination of G0's link", fontsize=8)
ax2.set_ylabel("wavelengths on link", fontsize=8.5)
ax2.set_title("(c) Source G0: λ per outgoing link", fontsize=9.5, color=INK, loc="left")
ax2.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
ax2.grid(True, axis="y", color=GRID, linewidth=0.6)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(colors=SEC_INK, labelsize=7.5)

# (d) receive rings on resonance per GPU, before vs after
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)
xs = range(N)
ax3.bar(list(xs), rx_after, width=0.7, color=[ORANGE if loadf[g] >= 1.5 else BLUE for g in range(N)])
ax3.axhline(384, color=MUTED, linewidth=1.2, label="before: 384 rings on, all GPUs")
ax3.axhline(max(rx_after), color=SEC_INK, linewidth=0.8, linestyle="--")
ax3.text(-0.4, max(rx_after) + 12, f"max {max(rx_after)} = {max(rx_after)/384:.2f}× uniform", fontsize=6.5,
         color=SEC_INK, ha="left")
ax3.set_xticks(list(xs))
ax3.set_xticklabels([str(g) for g in range(N)], fontsize=6.5)
ax3.set_xlabel("destination GPU", fontsize=8)
ax3.set_ylabel("Rx rings on resonance", fontsize=8.5)
ax3.set_title("(d) After: active drop rings per GPU", fontsize=9.5, color=INK, loc="left")
ax3.set_ylim(0, 660)
ax3.legend(frameon=False, fontsize=7, loc="upper right")
ax3.grid(True, axis="y", color=GRID, linewidth=0.6)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(colors=SEC_INK, labelsize=7.5)

fig.savefig("fig_lambda_reassign.png", dpi=300, facecolor=BG, bbox_inches="tight")
fig.savefig("fig_lambda_reassign.pdf", facecolor=BG, bbox_inches="tight")
print("wrote fig_lambda_reassign.png/.pdf")
print("source G0 after:", after, "| Rx rings max", max(rx_after), f"({max(rx_after)/384:.2f}x)")
