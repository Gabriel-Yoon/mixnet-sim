#!/usr/bin/env python3
"""Reduced-order thermal model (ROM) of the ANSYS OIO3D stack, and a full-depth stall estimate.

The ANSYS model is linear (constant properties, convective boundary), so the aPIC temperature
is a convolution of the XPU power with a fixed kernel. We identify that kernel as a sum of
first-order lags with fixed log-spaced time constants and non-negative gains, fitted on ONE
(schedule, trace) pair and validated on the others. The ROM then predicts the trace for a
synthetic full-depth schedule (each pipeline stage holds L/PP layers instead of one), which the
packet-level simulation could not produce, and the stall model of stall_model_v2.py is applied.

Usage: python3 rom_fulldepth.py
"""
import bisect
import csv
import json

import numpy as np

from derive_stall_physical import parse_prvar, DLAM_DT_PM_PER_K
from stall_model_v2 import (track, per_a2a_round_stall, load_a2a_windows, EPS_MAX_PM,
                            R_CTRL_BASE, STARTUP_SKIP_S)

DT = 1e-3
TAUS = np.array([2e-3, 5e-3, 12e-3, 30e-3, 80e-3, 0.2, 0.5, 1.2, 3.0, 8.0])

PAIRS = {
    "mixtral_median": ("mixtral_power_schedule_port800_median.csv", "mixtral_pic_transient_port800_median_dt1ms.csv"),
    "mixtral_cold": ("mixtral_power_schedule_port800_cold.csv", "mixtral_pic_transient_port800_cold_dt1ms.csv"),
    "llama_median": ("llama_power_schedule_port800_median.csv", "llama_pic_transient_port800_median_dt1ms.csv"),
    "llama_cold": ("llama_power_schedule_port800_cold.csv", "llama_pic_transient_port800_cold_dt1ms.csv"),
    "qwen_median": ("qwen_power_schedule_i1800_median.csv", "qwen_pic_transient_i1800_median_dt1ms.csv"),
    "qwen_cold": ("qwen_power_schedule_i1800_cold.csv", "qwen_pic_transient_i1800_cold_dt1ms.csv"),
}
# full model depth / pipeline stages -> layers held by one device
LAYERS_PER_STAGE = {"mixtral": 32 // 4, "llama": 32 // 4, "qwen": 24 // 4}


def load_sched(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["start_s"]), float(r["end_s"]), r["phase"], float(r["power_w"]), int(r["iteration"])))
    return rows


def power_grid(rows, n):
    p = np.full(n, rows[-1][3])
    for s, e, _, w, _ in rows:
        i0, i1 = int(round(s / DT)), int(round(e / DT))
        p[i0:min(i1, n)] = w
    return p


def lags(u):
    """States of the first-order lags driven by u (zero initial state)."""
    X = np.zeros((len(u), len(TAUS)))
    a = np.exp(-DT / TAUS)
    x = np.zeros(len(TAUS))
    for i in range(len(u)):
        x = a * x + (1 - a) * u[i]
        X[i] = x
    return X


def nnls(A, b, tol=1e-9):
    """Lawson-Hanson active-set non-negative least squares."""
    n = A.shape[1]
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(10 * n):
        if P.all() or w[~P].max() <= tol:
            break
        j = np.argmax(np.where(~P, w, -np.inf))
        P[j] = True
        while True:
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if s[P].min() > 0:
                break
            neg = P & (s <= 0)
            alpha = np.min(x[neg] / (x[neg] - s[neg]))
            x = x + alpha * (s - x)
            P &= x > tol
        x = s
        w = A.T @ (b - A @ x)
    return x


def load_pair(key):
    sp, tp = PAIRS[key]
    rows = load_sched(sp)
    t, T = parse_prvar(tp)
    t, T = np.array(t), np.array(T)
    n = int(round(min(t[-1], rows[-1][1]) / DT))
    tg = np.arange(1, n + 1) * DT
    Tg = np.interp(tg, t, T)
    return rows, tg, Tg, power_grid(rows, n), T[0]


P_IDLE, P_FULL = 84.0, 700.0


def design(p):
    # the traces start from the idle steady state, so only the power above idle drives the lags
    return lags((p - P_IDLE) / (P_FULL - P_IDLE))


def fit(key):
    rows, tg, Tg, p, T0 = load_pair(key)
    g = nnls(design(p), Tg - T0)
    return g


def predict(g, p, T0):
    return T0 + design(p) @ g


def stall_stats(tg, Tg, wins):
    t = list(tg)
    dlam = [(x - Tg[0]) * DLAM_DT_PM_PER_K for x in Tg]
    eps = track(t, dlam, R_CTRL_BASE * DLAM_DT_PM_PER_K)
    rs = per_a2a_round_stall(t, eps, EPS_MAX_PM, wins, DT)
    v = [x for x, _, _ in rs]
    by = {}
    for x, info, _ in rs:
        by.setdefault(info, []).append(x)
    return float(np.mean(v)) if v else float("nan"), {k: float(np.mean(x)) for k, x in by.items()}, len(v)


def swing(tg, Tg):
    m = tg >= STARTUP_SKIP_S
    return float(Tg[m].max() - Tg[m].min()), float(Tg[m].mean())


def full_depth(rows, wins, k, n_rep=6):
    """Synthetic full-depth schedule for one device. Within the proxy iteration the device's work
    comes in blocks separated by long waits (other pipeline stages / the other pass). With k layers
    per stage every block of busy work is executed k times back to back, and every long wait (>= the
    LONG threshold) is k times longer because the other stages also hold k layers."""
    it1 = [r for r in rows if r[4] == 1]
    t0 = it1[0][0]
    period = it1[-1][1] - t0
    LONG = 0.030
    w1 = [(s - t0, e - t0, info) for s, e, info, it in wins if it == 1]
    # segment the iteration into busy blocks and long waits
    segs, cur = [], []
    def own_round_wait(s, e):
        ov = sum(max(0.0, min(e, b + t0) - max(s, a + t0)) for a, b, _ in w1)
        return ov >= 0.5 * (e - s)

    for s, e, ph, w, _ in it1:
        d = e - s
        if ph == "idle" and d >= LONG and not own_round_wait(s, e):
            if cur:
                segs.append(("busy", cur))
                cur = []
            segs.append(("wait", [(s - t0, e - t0, ph, w)]))
        else:
            cur.append((s - t0, e - t0, ph, w))
    if cur:
        segs.append(("busy", cur))
    out, owins, clock = [], [], 0.0
    for kind, items in segs:
        a, b = items[0][0], items[-1][1]
        if kind == "wait":
            d = (b - a) * k
            out.append((clock, clock + d, "idle", items[0][3]))
            # rounds of other devices' layers that fall in a wait are not this device's rounds
            clock += d
        else:
            for rep in range(k):
                for s, e, ph, w in items:
                    out.append((clock + s - a, clock + e - a, ph, w))
                for s, e, info in w1:
                    if a - 1e-9 <= s < b + 1e-9:
                        owins.append((clock + s - a, clock + e - a, info))
                clock += b - a
    per = clock
    rows_fd, wins_fd = [], []
    for it in range(n_rep):
        for s, e, ph, w in out:
            rows_fd.append((s + it * per, e + it * per, ph, w, it + 1))
        for s, e, info in owins:
            wins_fd.append((s + it * per, e + it * per, info, it + 1))
    return rows_fd, wins_fd, per, period


def main():
    res = {}
    g = fit("mixtral_median")
    print("ROM fitted on mixtral_median; gains by tau:")
    for tau, gp in zip(TAUS, g):
        print(f"  tau {tau*1e3:8.0f} ms  gain {gp:.3f} K at full power")
    print("\nvalidation (ROM from mixtral_median applied to every schedule):")
    res["validation"] = {}
    for key in PAIRS:
        rows, tg, Tg, p, T0 = load_pair(key)
        Tp = predict(g, p, T0)
        m = tg >= STARTUP_SKIP_S
        rmse = float(np.sqrt(np.mean((Tp[m] - Tg[m]) ** 2)))
        sw_a, mean_a = swing(tg, Tg)
        sw_r, mean_r = swing(tg, Tp)
        wins = load_a2a_windows(PAIRS[key][0].replace(".csv", "_a2a_windows.csv"))
        st_a, _, n = stall_stats(tg, Tg, wins)
        st_r, _, _ = stall_stats(tg, Tp, wins)
        print(f"  {key:15s} rmse {rmse:.2f} K | swing ANSYS {sw_a:.1f} ROM {sw_r:.1f} K | mean {mean_a:.1f} / {mean_r:.1f} C"
              f" | stall/round ANSYS {st_a:.1f} ROM {st_r:.1f} ms ({n} rounds)")
        res["validation"][key] = dict(rmse_K=rmse, swing_ansys=sw_a, swing_rom=sw_r, stall_ansys_ms=st_a, stall_rom_ms=st_r)

    print("\nfull-depth estimate (k layers per pipeline stage):")
    res["full_depth"] = {}
    for key in PAIRS:
        model = key.split("_")[0]
        k = LAYERS_PER_STAGE[model]
        rows, tg, Tg, p, T0 = load_pair(key)
        wins = load_a2a_windows(PAIRS[key][0].replace(".csv", "_a2a_windows.csv"))
        Tp1 = predict(g, p, T0)
        st1, by1, n1 = stall_stats(tg, Tp1, wins)
        rows_fd, wins_fd, per_fd, per1 = full_depth(rows, wins, k)
        n = int(round(rows_fd[-1][1] / DT))
        pfd = power_grid(rows_fd, n)
        tfd = np.arange(1, n + 1) * DT
        Tfd = predict(g, pfd, T0)
        # skip the first two synthetic iterations (start-up + slow package lag)
        skip = 2 * per_fd
        wins_ss = [w for w in wins_fd if w[0] >= skip]
        stk, byk, nk = stall_stats(tfd, Tfd, wins_ss)
        m = tfd >= skip
        duty1 = float(np.mean(p > 100))
        dutyk = float(np.mean(pfd > 100))
        print(f"  {key:15s} k={k}: period {per1*1e3:.0f} -> {per_fd*1e3:.0f} ms, duty {duty1:.2f} -> {dutyk:.2f}, "
              f"swing {swing(tg, Tp1)[0]:.1f} -> {float(Tfd[m].max()-Tfd[m].min()):.1f} K, "
              f"stall/round (ROM) {st1:.1f} -> {stk:.1f} ms  [{n1//10} -> {nk//4} rounds/iter]")
        print("      by type: " + ", ".join(f"{a}: {by1.get(a, float('nan')):.1f}->{b:.1f}" for a, b in byk.items()))
        res["full_depth"][key] = dict(k=k, period_proxy_s=per1, period_full_s=per_fd, stall_proxy_ms=st1,
                                      stall_full_ms=stk, by_type_full=byk, by_type_proxy=by1,
                                      swing_full_K=float(Tfd[m].max() - Tfd[m].min()))
    with open("rom_fulldepth.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote rom_fulldepth.json")


if __name__ == "__main__":
    main()
