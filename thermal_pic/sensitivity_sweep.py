#!/usr/bin/env python3
"""One-at-a-time sensitivity sweep of the optical-domain stall derivation's
physical constants (ring Q, acceptable-detuning budget fraction of FWHM, and
controller tracking rate R_ctrl), re-using the already-computed ANSYS T(t)
traces (no new PACE/ANSYS runs needed -- only the downstream Delta-lambda ->
residual -> stall-window chain is re-evaluated in Python for each parameter
value).

Baseline: dlambda/dT=80 pm/K, Q=8000, eps_max=10% FWHM, R_ctrl=0.0625 K/ms
(all as in derive_stall_physical.py).
"""
import csv
from derive_stall_physical import parse_prvar, DLAM_DT_PM_PER_K

LAMBDA_NM = 1550.0


def simulate_residual(t, T, r_ctrl_pm_per_ms):
    T0 = T[0]
    dlam = [(Ti - T0) * DLAM_DT_PM_PER_K for Ti in T]
    lam_track = [0.0] * len(t)
    for i in range(1, len(t)):
        dt_ms = (t[i] - t[i - 1]) * 1000.0
        if dt_ms <= 0:
            lam_track[i] = lam_track[i - 1]
            continue
        desired = dlam[i] - lam_track[i - 1]
        max_step = r_ctrl_pm_per_ms * dt_ms
        step = max(-max_step, min(max_step, desired))
        lam_track[i] = lam_track[i - 1] + step
    eps = [abs(dlam[i] - lam_track[i]) for i in range(len(t))]
    return eps, dlam


def find_stall_windows(t, eps, eps_max):
    windows = []
    in_stall = False
    start = None
    for i in range(len(t)):
        if eps[i] > eps_max and not in_stall:
            in_stall = True
            start = t[i]
        elif eps[i] <= eps_max and in_stall:
            in_stall = False
            windows.append((t[i] - start) * 1000.0)
    if in_stall:
        windows.append((t[-1] - start) * 1000.0)
    return windows


def steady_state_mean(t, T, q, eps_max_frac, r_ctrl_k_per_ms):
    r_ctrl_pm_per_ms = r_ctrl_k_per_ms * DLAM_DT_PM_PER_K
    fwhm_pm = (LAMBDA_NM * 1000.0) / q
    eps_max_pm = eps_max_frac * fwhm_pm
    eps, dlam = simulate_residual(t, T, r_ctrl_pm_per_ms)
    durations = [d for d in find_stall_windows(t, eps, eps_max_pm) if d > 1.0]
    if len(durations) < 2:
        return None, len(durations)
    steady = durations[1:]  # drop startup transient
    return sum(steady) / len(steady), len(steady)


MODELS = [
    ("Mixtral-8x7B", "mixtral8x7b_pic_transient_L4seq4096.csv"),
    ("Qwen-MoE-14.3B", "qwenmoe_pic_transient_L4seq4096.csv"),
    ("LLaMA-MoE-6.7B", "llamamoe_pic_transient_L4seq4096.csv"),
]

BASE_Q = 8000.0
BASE_EPS_FRAC = 0.10
BASE_RCTRL = 0.0625

Q_SWEEP = [5000, 8000, 12000, 15000]
EPS_FRAC_SWEEP = [0.05, 0.10, 0.15, 0.20]
RCTRL_SWEEP = [0.03, 0.0625, 0.10, 0.15]

if __name__ == "__main__":
    traces = {}
    for name, path in MODELS:
        traces[name] = parse_prvar(path)

    rows = []
    print("=== Q sweep (eps_max_frac=10%, R_ctrl=0.0625 K/ms fixed) ===")
    for q in Q_SWEEP:
        for name, _ in MODELS:
            t, T = traces[name]
            mean, n = steady_state_mean(t, T, q, BASE_EPS_FRAC, BASE_RCTRL)
            print(f"  Q={q:6.0f}  {name:18s} mean={mean:6.1f} ms  n={n}")
            rows.append(("Q", q, name, mean, n))

    print("\n=== eps_max fraction sweep (Q=8000, R_ctrl=0.0625 K/ms fixed) ===")
    for frac in EPS_FRAC_SWEEP:
        for name, _ in MODELS:
            t, T = traces[name]
            mean, n = steady_state_mean(t, T, BASE_Q, frac, BASE_RCTRL)
            print(f"  eps_max={frac*100:4.0f}%FWHM  {name:18s} mean={mean:6.1f} ms  n={n}")
            rows.append(("eps_max_frac", frac, name, mean, n))

    print("\n=== R_ctrl sweep (Q=8000, eps_max_frac=10% fixed) ===")
    for rc in RCTRL_SWEEP:
        for name, _ in MODELS:
            t, T = traces[name]
            mean, n = steady_state_mean(t, T, BASE_Q, BASE_EPS_FRAC, rc)
            print(f"  R_ctrl={rc:.4f} K/ms  {name:18s} mean={mean:6.1f} ms  n={n}")
            rows.append(("R_ctrl", rc, name, mean, n))

    with open("sensitivity_sweep_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "model", "mean_stall_ms", "n_windows"])
        for r in rows:
            w.writerow(r)
    print("\nwrote sensitivity_sweep_results.csv")
