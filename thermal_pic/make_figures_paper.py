#!/usr/bin/env python3
"""Five publication figures for the revision, one claim each (run from thermal_pic/):

  F1 fig_mechanism     thermal transient -> residual detuning -> rounds caught in stall windows
  F2 fig_headline      CPO port-rate sweep vs fat-tree; stall-free vs thermo-optic at the design point
  F3 fig_mitigation    per-round delay vs loop tracking rate and vs feed-forward timing error
  F4 fig_calibration   H100 telemetry (two timescales) and swing vs burst length, measured vs ANSYS
  F5 fig_reassignment  wavelength reassignment before/after on one wafer; gain by fabric regime

Vector PDF + PNG in figs_paper/. Fonts: Linux Libertine (acmart body font) from the tectonic cache.
"""
import ast
import bisect
import csv
import glob
import gzip
import os
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

from derive_stall_physical import parse_prvar
from stall_model_v2 import (resample, track, load_schedule, load_a2a_windows, per_a2a_round_stall,
                            jittered_prediction_targets, DLAM_DT_PM_PER_K, EPS_MAX_PM, R_CTRL_BASE)

# ---------------------------------------------------------------- style
for p in glob.glob(os.path.expanduser("~/Library/Caches/Tectonic/bundles/data/*/LinLibertine_*.otf")):
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
_names = {f.name for f in fm.fontManager.ttflist}
FONT = "Linux Libertine O" if "Linux Libertine O" in _names else ("STIX Two Text" if "STIX Two Text" in _names else "serif")
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "mathtext.fontset": "stix",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#008300"
MUTED, GRID, INK, SEC_INK, BG, BUSY = "#898781", "#e6e5df", "#0b0b0b", "#52514e", "white", "#ebeae4"
OUT = "figs_paper"
os.makedirs(OUT, exist_ok=True)
DT = 0.001

MODELS = [  # label, color, trace, schedule, stall-free / stalled iteration (ms)
    ("Mixtral 8×7B", BLUE, "mixtral_pic_transient_port800_median_dt1ms.csv", "mixtral_power_schedule_port800_median.csv", 865.3, 1370.5),
    ("LLaMA-MoE 6.7B", GREEN, "llama_pic_transient_port800_cold_dt1ms.csv", "llama_power_schedule_port800_cold.csv", 494.4, 1395.6),
    ("Qwen-MoE 14.3B", ORANGE, "qwen_pic_transient_i1800_median_dt1ms.csv", "qwen_power_schedule_i1800_median.csv", 661.3, 1179.4),
]
FAT = {"Mixtral 8×7B": 841.3, "LLaMA-MoE 6.7B": 427.4, "Qwen-MoE 14.3B": 195.4}


def style(ax, grid="y"):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8ae")
    ax.tick_params(colors=INK, length=2.5)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def panel(ax, label, dx=-0.16, dy=1.04):
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", facecolor=BG, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", dpi=400, facecolor=BG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def load_model(m):
    label, color, trace, sched_path, *_ = m
    t_raw, T_raw = parse_prvar(trace)
    t, T = resample(t_raw, T_raw, DT)
    sched = load_schedule(sched_path)
    wins = load_a2a_windows(sched_path.replace(".csv", "_a2a_windows.csv"))
    dlam = [(Ti - T[0]) * DLAM_DT_PM_PER_K for Ti in T]
    eps = track(t, dlam, R_CTRL_BASE * DLAM_DT_PM_PER_K)
    return dict(label=label, color=color, t=t, T=T, sched=sched, wins=wins, dlam=dlam, eps=eps,
                period=sched[-1][1] / 10.0)


# ================================================================ F1 mechanism
def fig_mechanism(data):
    d = data[1]  # LLaMA-MoE: the worst-hit model
    fig = plt.figure(figsize=(6.8, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.45, wspace=0.32,
                          left=0.08, right=0.99, top=0.93, bottom=0.14)
    t0 = 3 * d["period"] + 0.045
    t1 = t0 + 0.30
    i0, i1 = bisect.bisect_left(d["t"], t0), bisect.bisect_left(d["t"], t1)
    xs = [(x - t0) * 1e3 for x in d["t"][i0:i1]]

    ax = fig.add_subplot(gs[0, 0])
    for s, e, ph, p in d["sched"]:
        if ph != "idle" and e > t0 and s < t1:
            ax.axvspan((max(s, t0) - t0) * 1e3, (min(e, t1) - t0) * 1e3, color=BUSY, linewidth=0)
    ax.plot(xs, d["T"][i0:i1], color=d["color"])
    style(ax)
    ax.set_xlim(0, 300)
    ax.set_ylabel("aPIC temperature (°C)")
    ax.tick_params(labelbottom=False)
    ax.text(0.015, 0.94, "shaded: device computing", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=SEC_INK)
    panel(ax, "(a)", dx=-0.13)

    ax = fig.add_subplot(gs[1, 0])
    ep = d["eps"][i0:i1]
    ax.fill_between(xs, 0, ep, where=[e > EPS_MAX_PM for e in ep], color=d["color"], alpha=0.16, linewidth=0)
    ax.plot(xs, ep, color=d["color"])
    ax.axhline(EPS_MAX_PM, color=INK, linewidth=0.7, linestyle=(0, (4, 3)))
    ready = [(s - t0) * 1e3 for s, e, info, it in d["wins"] if t0 <= s <= t1]
    ymax = max(ep) * 1.12
    ax.vlines(ready, 0, ymax * 0.10, color=INK, linewidth=0.7)
    ax.text(0.99, 0.94, "ticks: all-to-all rounds becoming ready\ndashed: detuning budget (10 % FWHM)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=SEC_INK, linespacing=1.3)
    style(ax)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("residual detuning ε (pm)")
    panel(ax, "(b)", dx=-0.13)

    ax = fig.add_subplot(gs[:, 1])
    for dd in data:
        vals = sorted(v for v, _, _ in per_a2a_round_stall(dd["t"], dd["eps"], EPS_MAX_PM, dd["wins"], DT))
        ys = [i / len(vals) for i in range(1, len(vals) + 1)]
        ax.step(vals, ys, where="post", color=dd["color"], label=f"{dd['label']}  ({statistics.mean(vals):.0f} ms mean)")
    style(ax, grid="both")
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("delay per all-to-all round (ms)")
    ax.set_ylabel("fraction of rounds")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    panel(ax, "(c)", dx=-0.2)
    save(fig, "fig_mechanism")


# ================================================================ F2 headline
def fig_headline():
    rates = [200, 400, 800, 1600]
    sweep = {"Mixtral 8×7B": [890.9, 873.7, 865.3, 863.9], "LLaMA-MoE 6.7B": [495.7, 490.4, 494.4, 491.4],
             "Qwen-MoE 14.3B": [807.4, 678.6, 661.3, 673.4]}
    gateway = {"Mixtral 8×7B": 1954.9, "LLaMA-MoE 6.7B": 493.1, "Qwen-MoE 14.3B": 1452.1}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.3,
                                                                              left=0.08, right=0.99, top=0.9, bottom=0.18))
    for label, color, *_ in MODELS:
        ax.plot(rates, sweep[label], color=color, marker="o", markersize=3.5, label=label)
        ax.axhline(FAT[label], color=color, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.scatter([120], [gateway[label]], color=color, marker="s", s=18, zorder=5)
    handles = [Line2D([], [], color=c, marker="o", markersize=3.5) for _, c, *_ in MODELS]
    handles += [Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=0.8),
                Line2D([], [], color=INK, marker="s", linestyle="none", markersize=4)]
    ax.legend(handles, [m[0] for m in MODELS] + ["ideal fat-tree, same injection", "single gateway per wafer pair"],
              frameon=False, loc="upper right", handlelength=1.8, ncol=1)
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([120] + rates)
    ax.set_xticklabels(["gw", "200", "400", "800", "1600"])
    ax.minorticks_off()
    ax.set_xlabel("off-wafer CPO port rate per reticle (GB/s)")
    ax.set_ylabel("stall-free iteration time (ms)")
    ax.set_ylim(0, 2100)
    panel(ax, "(a)", dx=-0.13)

    labels = [m[0] for m in MODELS]
    free, stall = [m[4] for m in MODELS], [m[5] for m in MODELS]
    xs = range(3)
    ax2.bar([x - 0.19 for x in xs], free, width=0.36, color=MUTED, label="athermal ring + programmable setpoint")
    ax2.bar([x + 0.19 for x in xs], stall, width=0.36, color=BLUE, label="thermo-optic tuning (stall)")
    ax2.scatter(list(xs), [FAT[l] for l in labels], marker="_", s=180, color=INK, linewidths=1.2, zorder=5, label="ideal fat-tree")
    for x, f, s in zip(xs, free, stall):
        ax2.text(x + 0.19, s + 25, f"{s/f:.2f}×", ha="center", fontsize=7.5, color=INK)
    style(ax2)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(["Mixtral\n8×7B", "LLaMA-MoE\n6.7B", "Qwen-MoE\n14.3B"])
    ax2.set_ylabel("iteration time (ms)")
    ax2.set_ylim(0, 2000)
    ax2.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.2)
    panel(ax2, "(b)", dx=-0.2)
    save(fig, "fig_headline")


# ================================================================ F3 mitigation
def fig_mitigation(data):
    rates = [0.02, 0.0625, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    jitters = [0, 1, 2, 5, 10, 20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw=dict(wspace=0.3, left=0.08, right=0.99, top=0.9, bottom=0.2))
    for d in data:
        ys = []
        for r in rates:
            eps = track(d["t"], d["dlam"], r * DLAM_DT_PM_PER_K)
            ys.append(statistics.mean(v for v, _, _ in per_a2a_round_stall(d["t"], eps, EPS_MAX_PM, d["wins"], DT)))
        ax1.plot(rates, ys, color=d["color"], marker="o", markersize=3.5, label=d["label"])
        yj = []
        for j in jitters:
            ff = jittered_prediction_targets(d["t"], d["dlam"], d["sched"], DT, float(j))
            eps = track(d["t"], d["dlam"], R_CTRL_BASE * DLAM_DT_PM_PER_K, feedforward=ff)
            yj.append(statistics.mean(v for v, _, _ in per_a2a_round_stall(d["t"], eps, EPS_MAX_PM, d["wins"], DT)))
        ax2.plot(jitters, yj, color=d["color"], marker="o", markersize=3.5)
    ax1.axvline(R_CTRL_BASE, color=INK, linewidth=0.7, linestyle=(0, (4, 3)))
    ax1.text(R_CTRL_BASE * 1.12, 44, "0.0625 K/ms\n(this work)", fontsize=6.8, color=SEC_INK, va="top")
    style(ax1)
    ax1.set_xscale("log")
    ax1.set_xlabel("tracking-loop slew rate $R_\\mathrm{ctrl}$ (K/ms)")
    ax1.set_ylabel("mean delay per a2a round (ms)")
    ax1.set_ylim(0, 46)
    ax1.legend(frameon=False, loc="upper right", handlelength=1.6)
    panel(ax1, "(a)", dx=-0.15)
    style(ax2)
    ax2.set_xlabel("timing error of schedule-based feed-forward (ms)")
    ax2.set_ylabel("mean delay per a2a round (ms)")
    ax2.set_ylim(0, 30)
    ax2.set_xticks(jitters)
    panel(ax2, "(b)", dx=-0.15)
    save(fig, "fig_mitigation")


# ================================================================ F4 calibration
def fig_calibration():
    tel = [(float(r["wall_time_s"]), int(r["temp_gpu_C"]))
           for r in csv.DictReader(gzip.open("telemetry/synth/telemetry_20ms.csv.gz", "rt")) if r["nvml_index"] == "0"]
    ph = list(csv.DictReader(open("telemetry/synth/phases.csv")))
    t0 = tel[0][0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.3,
                                                                              left=0.08, right=0.99, top=0.9, bottom=0.2))
    ax.plot([x - t0 for x, _ in tel], [T for _, T in tel], color=BLUE, linewidth=0.9)
    names = {"a_35on_350off": "35 / 350 ms", "b_35on_35off": "35 / 35 ms", "c_300on_350off": "300 / 350 ms", "d_60son_60soff": "60 s / 60 s"}
    for k, (pat, nm) in enumerate(names.items()):
        rows = [r for r in ph if r["pattern"] == pat]
        s = min(float(r["start_s"]) for r in rows) - t0
        e = max(float(r["end_s"]) for r in rows) - t0
        ax.axvspan(s, e, color=BUSY if k % 2 == 0 else "#f5f4ef", linewidth=0)
        ax.text((s + e) / 2, 74.5, nm, fontsize=6.5, color=SEC_INK, ha="center", rotation=0 if k == 3 else 30)
    style(ax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("H100 die temperature (°C)")
    ax.set_ylim(30, 80)
    panel(ax, "(a)", dx=-0.12)

    ax2.scatter([35, 300, 60000], [4, 13, 34], color=BLUE, s=26, zorder=5, label="H100 die sensor, measured")
    for x, y, c in zip([4.2, 25, 52], [3.3, 9.4, 11.4], [ORANGE, GREEN, BLUE]):
        ax2.scatter([x], [y], color=c, marker="D", s=24, zorder=5, edgecolor=INK, linewidth=0.5)
    ax2.scatter([], [], color="white", marker="D", edgecolor=INK, label="ANSYS aPIC, MoE compute bursts")
    style(ax2, grid="both")
    ax2.set_xscale("log")
    ax2.set_xlabel("heating-burst length (ms)")
    ax2.set_ylabel("temperature swing (K)")
    ax2.set_ylim(0, 40)
    ax2.legend(frameon=False, loc="upper left", handlelength=1.2)
    panel(ax2, "(b)", dx=-0.2)
    save(fig, "fig_calibration")


# ================================================================ F5 reassignment
def _alloc_one_wafer():
    W = ast.literal_eval(open("../mixnet-htsim/test/wm_ep16.txt").read().strip())
    N, COLS = 16, 4
    row, col = (lambda g: g // COLS), (lambda g: g % COLS)

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

    def alloc(u, LAM=192, FLOOR=0.1, CAP=2.0):
        res = {}
        for grp in ([v for v in range(N) if v != u and row(v) == row(u)], [v for v in range(N) if v != u and col(v) == col(u)]):
            s = sum(load[u][v] for v in grp)
            w = {v: max(load[u][v], FLOOR * s / len(grp)) for v in grp}
            ws = sum(w.values())
            lam = {v: LAM * w[v] / ws for v in grp}
            cap = CAP * LAM / len(grp)
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
    colsum = [sum(W[i][j] for i in range(N)) for j in range(N)]
    loadf = [c / (sum(colsum) / N) for c in colsum]
    return alloc(0), loadf, row, col


def fig_reassignment():
    after, loadf, row, col = _alloc_one_wafer()
    fig = plt.figure(figsize=(6.8, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22, left=0.03, right=0.99, top=0.9, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.6, 9.2)
    ax.set_ylim(-1.15, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")

    def grid(xoff, lam, title):
        sx, sy = xoff + col(0), 3 - row(0)
        for v, l in lam.items():
            x, y = xoff + col(v), 3 - row(v)
            k = abs(x - sx) + abs(y - sy)
            off = 0.1 * (k - 2)
            p0, p1 = ((sx, sy + off), (x, y + off)) if col(v) != 0 else ((sx + off, sy), (x + off, y))
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, shrinkA=8, shrinkB=8,
                                         linewidth=0.4 + 2.6 * l / 128, color=BLUE, alpha=0.75, zorder=1))
        for g in range(16):
            x, y = xoff + col(g), 3 - row(g)
            hot = loadf[g] >= 1.5
            fc = ORANGE if hot else (BLUE if g == 0 else "#dcdbd3")
            ax.add_patch(Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, facecolor=fc, edgecolor="none", zorder=2))
            lab = f"G{g}\n{lam[g]} λ" if g in lam else f"G{g}"
            ax.text(x, y, lab, ha="center", va="center", fontsize=5.6, zorder=3, color="white" if (hot or g == 0) else SEC_INK)
        ax.text(xoff + 1.5, 3.62, title, ha="center", va="bottom", fontsize=7.5, color=INK)

    grid(0, {v: 64 for v in after}, "uniform: 64 λ per link")
    grid(5, after, "demand-aware (floor 0.1, cap 2×)")
    ax.text(4.3, -0.95, "orange: hot-expert reticles (load ≥ 1.5×)   blue: source G0   arrow width ∝ λ",
            ha="center", fontsize=6.3, color=SEC_INK)
    panel(ax, "(a)", dx=0.0, dy=0.97)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["Mixtral\n8×7B", "LLaMA-MoE\n6.7B", "Qwen-MoE\n14.3B"]
    mesh = [-36.0, -9.0, -10.4]
    port = [-3.4, -10.5, None]
    port_stall = [-2.9, 0.9, 1.3]
    xs = range(3)
    ax2.bar([x - 0.27 for x in xs], mesh, width=0.25, color=MUTED, label="mesh/gateway-bound fabric")
    ax2.bar([x for x in xs], [v or 0 for v in port], width=0.25, color=BLUE, label="CPO ports 800 GB/s, stall-free")
    ax2.bar([x + 0.27 for x in xs], port_stall, width=0.25, color=BLUE, hatch="////", edgecolor="white", linewidth=0,
            label="CPO ports 800 GB/s + thermo-optic stall")
    for x, vals in zip(xs, zip(mesh, port, port_stall)):
        for dx, v in zip((-0.27, 0, 0.27), vals):
            if v is None:
                ax2.text(x + dx, 0.8, "n/a", ha="center", fontsize=6, color=SEC_INK)
            else:
                ax2.text(x + dx, v - 2.0 if v < 0 else v + 0.8, f"{v:+.0f}%", ha="center", va="top" if v < 0 else "bottom", fontsize=6.3, color=INK)
    ax2.axhline(0, color=SEC_INK, linewidth=0.6)
    style(ax2)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("iteration-time change vs uniform λ (%)")
    ax2.set_ylim(-43, 12)
    ax2.legend(frameon=False, loc="lower right", handlelength=1.3, fontsize=6.3)
    panel(ax2, "(b)", dx=-0.22, dy=0.97)
    save(fig, "fig_reassignment")


if __name__ == "__main__":
    print("font:", FONT)
    data = [load_model(m) for m in MODELS]
    fig_mechanism(data)
    fig_headline()
    fig_mitigation(data)
    fig_calibration()
    fig_reassignment()
