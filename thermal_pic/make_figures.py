#!/usr/bin/env python3
"""Regenerate every revision figure from the final data (run from thermal_pic/).

Outputs figs_final/<name>.{png,pdf}. Data sources: 1 ms ANSYS traces on the simulated per-device
timelines (port-800 design point; qwen on its F1 timeline), the per-device a2a round windows,
results_consolidated.md numbers (typed in below), and the H100 telemetry (Job C).
"""
import bisect
import csv
import gzip
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from derive_stall_physical import parse_prvar
from stall_model_v2 import (resample, track, load_schedule, load_a2a_windows, per_a2a_round_stall,
                            jittered_prediction_targets, DLAM_DT_PM_PER_K, EPS_MAX_PM, FWHM_PM,
                            R_CTRL_BASE, STARTUP_SKIP_S)

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#008300"
MUTED, GRID, INK, SEC_INK, BG, BUSY = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe5"
OUT = "figs_final"
os.makedirs(OUT, exist_ok=True)

MODELS = [  # label, color, trace, schedule, iteration length (ms), stall-free/stalled makespans (ms)
    ("Mixtral 8×7B", BLUE, "mixtral_pic_transient_port800_median_dt1ms.csv",
     "mixtral_power_schedule_port800_median.csv", 865.3, 1370.5),
    ("LLaMA-MoE 6.7B", GREEN, "llama_pic_transient_port800_cold_dt1ms.csv",
     "llama_power_schedule_port800_cold.csv", 494.4, 1395.6),
    ("Qwen-MoE 14.3B", ORANGE, "qwen_pic_transient_i1800_median_dt1ms.csv",
     "qwen_power_schedule_i1800_median.csv", 661.3, 1179.4),
]
DT = 0.001


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=SEC_INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, facecolor=BG, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", facecolor=BG, bbox_inches="tight")
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
    period = sched[-1][1] / 10.0
    return dict(label=label, color=color, t=t, T=T, sched=sched, wins=wins, dlam=dlam, eps=eps, period=period)


def busy_spans(sched, t0, t1):
    return [(max(s, t0), min(e, t1)) for s, e, ph, p in sched if ph != "idle" and e > t0 and s < t1]


# ---------------------------------------------------------------- Fig A: aPIC temperature, one iteration
def fig_transient(data):
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 5.4))
    fig.patch.set_facecolor(BG)
    for ax, d in zip(axes, data):
        t0 = 3 * d["period"]
        t1 = 4 * d["period"]
        for s, e in busy_spans(d["sched"], t0, t1):
            ax.axvspan((s - t0) * 1e3, (e - t0) * 1e3, color=BUSY, linewidth=0)
        i0, i1 = bisect.bisect_left(d["t"], t0), bisect.bisect_left(d["t"], t1)
        xs = [(x - t0) * 1e3 for x in d["t"][i0:i1]]
        ys = d["T"][i0:i1]
        ax.plot(xs, ys, color=d["color"], linewidth=1.6)
        ax.set_title(f"{d['label']} — swing {max(ys)-min(ys):.1f} K, mean {statistics.mean(ys):.1f} °C",
                     loc="left", fontsize=8.5, color=INK)
        style(ax)
        ax.set_xlim(0, (t1 - t0) * 1e3)
        ax.set_ylabel("aPIC T (°C)", fontsize=8.5)
    axes[-1].set_xlabel("time within one training iteration (ms)", fontsize=9)
    fig.text(0.5, -0.005, "shaded = device computing (GEMMs); unshaded = waiting on all-to-all / pipeline",
             ha="center", fontsize=7, color=SEC_INK)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_pic_mrr_transient")


# ---------------------------------------------------------------- Fig B: residual, rounds, per-round delay
def fig_stall(data):
    fig = plt.figure(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.6, wspace=0.55)
    for k, d in enumerate(data[:2]):
        ax = fig.add_subplot(gs[k, 0])
        t0 = 3 * d["period"]
        t1 = t0 + min(0.35, d["period"])
        i0, i1 = bisect.bisect_left(d["t"], t0), bisect.bisect_left(d["t"], t1)
        xs = [(x - t0) * 1e3 for x in d["t"][i0:i1]]
        ep = d["eps"][i0:i1]
        ax.fill_between(xs, 0, ep, where=[e > EPS_MAX_PM for e in ep], color=d["color"], alpha=0.18, linewidth=0)
        ax.plot(xs, ep, color=d["color"], linewidth=1.4)
        ax.axhline(EPS_MAX_PM, color=SEC_INK, linewidth=0.8, linestyle="--")
        ax.text(1, EPS_MAX_PM * 3.2, "budget 10 % FWHM", fontsize=7, color=SEC_INK, ha="left")
        # a2a rounds of this device's layer becoming ready inside the window
        ready = [(s - t0) * 1e3 for s, e, info, it in d["wins"] if t0 <= s <= t1]
        ymax = max(ep) * 1.15 if max(ep) > 0 else 100
        ax.vlines(ready, 0, ymax * 0.12, color=INK, linewidth=0.6)
        ax.set_title(f"{d['label']} — residual ε(t); ticks = rounds ready", loc="left",
                     fontsize=8, color=INK)
        style(ax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("ε (pm)", fontsize=8.5)
        if k == 1:
            ax.set_xlabel("time (ms)", fontsize=9)
    ax = fig.add_subplot(gs[:, 1])
    for d in data:
        vals = sorted(v for v, _, _ in per_a2a_round_stall(d["t"], d["eps"], EPS_MAX_PM, d["wins"], DT))
        ys = [i / len(vals) for i in range(1, len(vals) + 1)]
        ax.plot(vals, ys, color=d["color"], linewidth=1.8, label=f"{d['label']} (mean {statistics.mean(vals):.0f} ms)")
    style(ax)
    ax.set_xlabel("delay per all-to-all round (ms)", fontsize=9)
    ax.set_ylabel("fraction of rounds", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.text(0.02, 0.97, "per-round delay = wait for the\nremaining stall at round-ready\n+ stall inside the round",
            transform=ax.transAxes, fontsize=7, color=SEC_INK, va="top")
    save(fig, "fig_stall_derivation")


# ---------------------------------------------------------------- Fig C: iteration time, stall-free vs stall
def fig_penalty():
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.patch.set_facecolor(BG)
    labels = [m[0] for m in MODELS]
    free = [m[4] for m in MODELS]
    stall = [m[5] for m in MODELS]
    fat = [841.3, 427.4, 195.4]
    xs = range(len(labels))
    ax.bar([x - 0.2 for x in xs], free, width=0.38, color=MUTED, label="stall-free (athermal ring + setpoint)")
    ax.bar([x + 0.2 for x in xs], stall, width=0.38, color=BLUE, label="thermo-optic tuning stall")
    for x, f, s in zip(xs, free, stall):
        ax.text(x - 0.2, f + 25, f"{f:.0f}", ha="center", fontsize=7.5, color=INK)
        ax.text(x + 0.2, s + 25, f"{s:.0f}  ({s/f:.2f}×)", ha="center", fontsize=7.5, color=INK)
    ax.scatter(list(xs), fat, marker="_", s=260, color=INK, linewidths=1.4, zorder=5, label="ideal non-blocking fat-tree")
    style(ax)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("iteration time (ms)", fontsize=9)
    ax.set_ylim(0, 1600)
    ax.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_iteration_time_stall")


# ---------------------------------------------------------------- Fig D: CPO port-rate sweep vs baselines
def fig_port_sweep():
    rates = [200, 400, 800, 1600]
    sweep = {"Mixtral 8×7B": [890.9, 873.7, 865.3, 863.9], "LLaMA-MoE 6.7B": [495.7, 490.4, 494.4, 491.4],
             "Qwen-MoE 14.3B": [807.4, 678.6, 661.3, 673.4]}
    gateway = {"Mixtral 8×7B": 1954.9, "LLaMA-MoE 6.7B": 493.1, "Qwen-MoE 14.3B": 1452.1}
    fat = {"Mixtral 8×7B": 841.3, "LLaMA-MoE 6.7B": 427.4, "Qwen-MoE 14.3B": 195.4}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    fig.patch.set_facecolor(BG)
    for label, color, *_ in MODELS:
        ax.plot(rates, sweep[label], color=color, marker="o", markersize=4.5, linewidth=1.8, label=label)
        ax.axhline(fat[label], color=color, linewidth=0.9, linestyle="--", alpha=0.8)
        ax.scatter([130], [gateway[label]], color=color, marker="s", s=28, zorder=5)
        ax.annotate(f"{sweep[label][-1]:.0f}", (rates[-1], sweep[label][-1]), xytext=(6, -3),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.text(135, 2000, "single 200 GB/s gateway\nper wafer pair (legacy)", fontsize=6.5, color=SEC_INK, va="bottom")
    style(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([130] + rates)
    ax.set_xticklabels(["gw", "200", "400", "800", "1600"], fontsize=8)
    ax.set_xlabel("per-reticle CPO port rate (GB/s)", fontsize=9)
    ax.set_ylabel("stall-free iteration time (ms)", fontsize=9)
    ax.set_ylim(0, 2200)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.text(0.02, 0.02, "dashed = ideal non-blocking fat-tree at the same 1.5 TB/s per-GPU injection",
            transform=ax.transAxes, fontsize=6.5, color=SEC_INK)
    save(fig, "fig_port_sweep")


# ---------------------------------------------------------------- Fig E: reassignment gain by regime
def fig_reassign_gain():
    labels = [m[0] for m in MODELS]
    mesh = [-36.0, -9.0, -10.4]       # gateway model, intra 256: best of expert/measured demand
    port = [-3.4, -10.5, None]        # port 800, stall-free
    port_stall = [-2.9, 0.9, 1.3]     # port 800 + thermo-optic stall
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.patch.set_facecolor(BG)
    xs = range(len(labels))
    ax.bar([x - 0.27 for x in xs], mesh, width=0.25, color=MUTED, label="mesh/gateway-bound fabric (256 GB/s intra, 200 GB/s gateway)")
    ax.bar([x for x in xs], [v if v is not None else 0 for v in port], width=0.25, color=BLUE, label="CPO port 800 GB/s, stall-free")
    ax.bar([x + 0.27 for x in xs], port_stall, width=0.25, color=BLUE, hatch="///", edgecolor=BG, label="CPO port 800 GB/s + thermo-optic stall")
    for x, vals in zip(xs, zip(mesh, port, port_stall)):
        for dx, v in zip((-0.27, 0, 0.27), vals):
            if v is None:
                ax.text(x + dx, 1.5, "n/a", ha="center", fontsize=6.5, color=SEC_INK)
            else:
                ax.text(x + dx, v - 2.2 if v < 0 else v + 0.8, f"{v:+.1f}%", ha="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=SEC_INK, linewidth=0.8)
    style(ax)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("iteration time change vs uniform λ (%)", fontsize=8.5)
    ax.set_ylim(-42, 14)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig_reassign_gain")


# ---------------------------------------------------------------- Fig F: H100 calibration
def fig_h100():
    tel = [(float(r["wall_time_s"]), int(r["temp_gpu_C"]), float(r["power_W"]))
           for r in csv.DictReader(gzip.open("telemetry/synth/telemetry_20ms.csv.gz", "rt")) if r["nvml_index"] == "0"]
    ph = list(csv.DictReader(open("telemetry/synth/phases.csv")))
    ts = [x[0] for x in tel]
    t0 = ts[0]
    fig = plt.figure(figsize=(7.2, 3.0))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([(x - t0) for x in ts], [x[1] for x in tel], color=BLUE, linewidth=0.9)
    names = {"a_35on_350off": "(a)", "b_35on_35off": "(b)", "c_300on_350off": "(c)", "d_60son_60soff": "(d) 60 s on / 60 s off"}
    for k, (pat, nm) in enumerate(names.items()):
        rows = [r for r in ph if r["pattern"] == pat]
        s = min(float(r["start_s"]) for r in rows) - t0
        e = max(float(r["end_s"]) for r in rows) - t0
        ax.axvspan(s, e, color=BUSY if k % 2 == 0 else "#f4f3ee", linewidth=0)
        ax.text((s + e) / 2, 73.5, nm, fontsize=6.5, color=SEC_INK, ha="center")
    style(ax)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("H100 GPU temperature (°C)", fontsize=8.5)
    ax.set_ylim(30, 76)
    ax.text(0.01, 0.04, "(a) 35 ms on / 350 ms off   (b) 35 / 35 ms   (c) 300 / 350 ms",
            transform=ax.transAxes, fontsize=6.5, color=SEC_INK)
    # panel b: measured vs modelled swing per burst length
    ax2 = fig.add_subplot(gs[0, 1])
    meas_x, meas_y = [35, 300, 60000], [4, 13, 34]
    ax2.scatter(meas_x, meas_y, color=BLUE, s=32, zorder=5, label="H100 die sensor (measured)")
    mod_x, mod_y, mod_c = [4.2, 25, 52], [3.3, 9.4, 11.4], [ORANGE, GREEN, BLUE]
    ax2.scatter(mod_x, mod_y, color=mod_c, marker="D", s=30, zorder=5, edgecolor=INK, linewidth=0.5)
    ax2.scatter([], [], color=BG, marker="D", edgecolor=INK, label="ANSYS aPIC, workload bursts (Qwen/LLaMA/Mixtral)")
    ax2.plot([1000, 1000], [0, 0], color=BG)
    style(ax2)
    ax2.set_xscale("log")
    ax2.set_xlabel("compute-burst length (ms)", fontsize=9)
    ax2.set_ylabel("temperature swing (K)", fontsize=8.5)
    ax2.set_ylim(0, 40)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "fig_h100_calibration")


# ---------------------------------------------------------------- Fig G: controller rate and feed-forward jitter
def fig_controller(data):
    rates = [0.02, 0.0625, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    jitters = [0, 1, 2, 5, 10, 20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.patch.set_facecolor(BG)
    for d in data:
        ys = []
        for r in rates:
            eps = track(d["t"], d["dlam"], r * DLAM_DT_PM_PER_K)
            vals = [v for v, _, _ in per_a2a_round_stall(d["t"], d["eps"] if False else eps, EPS_MAX_PM, d["wins"], DT)]
            ys.append(statistics.mean(vals))
        ax1.plot(rates, ys, color=d["color"], marker="o", markersize=4, linewidth=1.8, label=d["label"])
        yj = []
        for j in jitters:
            ff = jittered_prediction_targets(d["t"], d["dlam"], d["sched"], DT, j) if j > 0 else None
            eps = track(d["t"], d["dlam"], R_CTRL_BASE * DLAM_DT_PM_PER_K, feedforward=ff) if ff else \
                  track(d["t"], d["dlam"], R_CTRL_BASE * DLAM_DT_PM_PER_K,
                        feedforward=jittered_prediction_targets(d["t"], d["dlam"], d["sched"], DT, 0.0))
            vals = [v for v, _, _ in per_a2a_round_stall(d["t"], eps, EPS_MAX_PM, d["wins"], DT)]
            yj.append(statistics.mean(vals))
        ax2.plot(jitters, yj, color=d["color"], marker="o", markersize=4, linewidth=1.8, label=d["label"])
    ax1.axvline(R_CTRL_BASE, color=SEC_INK, linewidth=0.8, linestyle="--")
    ax1.text(R_CTRL_BASE * 1.1, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 30, "manuscript value\n0.0625 K/ms", fontsize=6.5, color=SEC_INK)
    style(ax1)
    ax1.set_xscale("log")
    ax1.set_xlabel("controller tracking rate R_ctrl (K/ms)", fontsize=9)
    ax1.set_ylabel("mean delay per a2a round (ms)", fontsize=8.5)
    ax1.legend(frameon=False, fontsize=7, loc="upper right")
    ax1.set_title("(a) feedback loop rate", fontsize=9, loc="left", color=INK)
    style(ax2)
    ax2.set_xlabel("timing error of schedule-based feed-forward (ms)", fontsize=8.5)
    ax2.set_ylabel("mean delay per a2a round (ms)", fontsize=8.5)
    ax2.set_title("(b) feed-forward at R_ctrl = 0.0625 K/ms", fontsize=9, loc="left", color=INK)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_controller")


# ---------------------------------------------------------------- Fig H: where the iteration goes (device view)
def fig_time_share(data):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    fig.patch.set_facecolor(BG)
    labels, comp, idle = [], [], []
    for d in data:
        rows = [r for r in d["sched"] if r[0] < d["period"]]
        c = sum(e - s for s, e, ph, p in rows if ph != "idle")
        i = sum(e - s for s, e, ph, p in rows if ph == "idle")
        labels.append(d["label"]); comp.append(100 * c / (c + i)); idle.append(100 * i / (c + i))
    labels.append("8×H100 measured\n(Mixtral-style, EP 8)")
    comp.append(56); idle.append(44)
    xs = range(len(labels))
    ax.barh(list(xs), comp, color=BLUE, label="device computing")
    ax.barh(list(xs), idle, left=comp, color=MUTED, label="waiting: all-to-all / pipeline")
    for x, c, i in zip(xs, comp, idle):
        if c >= 10:
            ax.text(c / 2, x, f"{c:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        else:
            ax.text(c + 1.2, x, f"{c:.0f}% computing", ha="left", va="center", fontsize=7, color=INK)
        ax.text(c + i / 2 + (8 if c < 10 else 0), x, f"{i:.0f}%", ha="center", va="center", fontsize=7.5, color=INK)
    style(ax)
    ax.set_yticks(list(xs))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of one training iteration (%)", fontsize=8.5)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    save(fig, "fig_time_share")


if __name__ == "__main__":
    data = [load_model(m) for m in MODELS]
    fig_transient(data)
    fig_stall(data)
    fig_penalty()
    fig_port_sweep()
    fig_reassign_gain()
    fig_h100()
    fig_controller(data)
    fig_time_share(data)
