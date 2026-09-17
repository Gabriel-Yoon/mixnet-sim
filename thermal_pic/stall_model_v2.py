#!/usr/bin/env python3
"""Extended optical-domain stall model on an ANSYS aPIC temperature trace T(t).

Adds to derive_stall_physical.py:
  1. closed-form stall estimate  (dlam_swing - eps_max) / R_ctrl  next to the simulated one
  2. controller slew-rate sweep over decades -> smallest R_ctrl with zero stall
  3. schedule-aware feed-forward controller (pre-biases toward the known steady-state
     detuning of the next phase; residual is then only the model error)
  4. per-all-to-all-round accounting: total stall time that overlaps a communication
     phase (the quantity to inject into htsim), not the mean window length
  5. graded Lorentzian penalty: effective link-rate factor 1/(1+(2 eps/FWHM)^2) and the
     time-averaged rate loss per round, as an alternative to a binary stall
  6. heater bias / free-spectral-range hop check: can a heater-only (heat-up) tuner cover
     the swing from its bias point, and what static tuning power that bias costs per die

Usage: python3 stall_model_v2.py [--dt-ms 0.5] [--traces L4seq4096|L4seq4096_dt1ms]
"""
import argparse
import bisect
import csv
import json
import math
import os

from derive_stall_physical import parse_prvar, DLAM_DT_PM_PER_K, LAMBDA_NM

MODELS = [
    ("Mixtral-8x7B", "mixtral8x7b", "mixtral8x7b_power_schedule_fwdbwd_{tag}.csv"),
    ("Qwen-MoE-14.3B", "qwenmoe", "qwen_moe_14.3b_power_schedule_fwdbwd_{tag}.csv"),
    ("LLaMA-MoE-6.7B", "llamamoe", "llama_moe_6.7b_power_schedule_fwdbwd_{tag}.csv"),
]

Q_FACTOR = 8000.0
FWHM_PM = LAMBDA_NM * 1000.0 / Q_FACTOR
EPS_MAX_PM = 0.10 * FWHM_PM
R_CTRL_BASE = 0.0625            # K/ms  (manuscript value, [PID])
R_SWEEP_K_PER_MS = [0.01, 0.02, 0.0625, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
STARTUP_SKIP_S = 1.6            # drop the ambient->steady startup transient

# heater / ring parameters (typical Si MRR, cited in-text when used)
FSR_NM = 10.0                   # ~10 um radius Si ring
HEATER_EFF_NM_PER_MW = 0.25     # thermo-optic tuning efficiency
RINGS_PER_DIE = 750             # Section 3.2 endpoint count


def resample(t, T, dt_s):
    tt, TT = [], []
    x = t[0]
    while x <= t[-1]:
        j = bisect.bisect_right(t, x)
        if j >= len(t):
            TT.append(T[-1])
        elif j == 0:
            TT.append(T[0])
        else:
            a = (x - t[j - 1]) / (t[j] - t[j - 1])
            TT.append(T[j - 1] + a * (T[j] - T[j - 1]))
        tt.append(x)
        x += dt_s
    return tt, TT


def track(t, dlam, r_pm_per_ms, feedforward=None):
    """Rate-limited feedback tracker. With `feedforward` (a predicted detuning per sample),
    the heater is pre-driven to the prediction (the actuator itself is us-fast, so this
    part is not slew-limited) and the slew-limited feedback loop only corrects the
    prediction error dlam - prediction."""
    n = len(t)
    pred = [0.0] * n if feedforward is None else feedforward
    corr = [0.0] * n
    for i in range(1, n):
        dt_ms = (t[i] - t[i - 1]) * 1000.0
        desired = (dlam[i] - pred[i]) - corr[i - 1]
        step = max(-r_pm_per_ms * dt_ms, min(r_pm_per_ms * dt_ms, desired))
        corr[i] = corr[i - 1] + step
    return [abs(dlam[i] - pred[i] - corr[i]) for i in range(n)]


def windows(t, eps, eps_max, tmin):
    out, start = [], None
    for i in range(len(t)):
        if eps[i] > eps_max and start is None:
            start = t[i]
        elif eps[i] <= eps_max and start is not None:
            if start >= tmin:
                out.append((start, t[i]))
            start = None
    if start is not None and start >= tmin:
        out.append((start, t[-1]))
    return out


def load_schedule(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["start_s"]), float(r["end_s"]), r["phase"], float(r["power_w"])))
    return rows


def phase_lookup(sched):
    starts = [s for s, _, _, _ in sched]

    def at(x):
        j = bisect.bisect_right(starts, x) - 1
        if j < 0:
            return sched[0][2]
        return sched[j][2] if x < sched[j][1] else "gap"
    return at


def periodic_prediction_targets(t, dlam, sched, dt_s, n_iters=10, model_error_frac=0.0):
    """Schedule-aware feed-forward target: the training loop is periodic, so the
    controller can pre-compensate with the detuning trajectory it observed one
    iteration earlier (target(t) = dlam(t - T_iter)). model_error_frac scales the
    prediction to emulate an imperfect thermal model (0 = perfect prediction)."""
    period_s = sched[-1][1] / n_iters
    n_p = int(round(period_s / dt_s))
    tgt = list(dlam)
    for i in range(n_p, len(dlam)):
        tgt[i] = dlam[i - n_p] * (1.0 - model_error_frac)
    return tgt


def jittered_prediction_targets(t, dlam, sched, dt_s, jitter_ms, n_iters=10):
    """Feed-forward with timing error: the predicted trajectory is the previous iteration's
    shifted by `jitter_ms` (all-to-all durations vary with routing, so the next burst
    does not land exactly where predicted). Residual during a burst ~ slope x jitter."""
    period_s = sched[-1][1] / n_iters
    n_p = int(round(period_s / dt_s))
    n_j = int(round(jitter_ms / 1000.0 / dt_s))
    tgt = list(dlam)
    for i in range(n_p + n_j, len(dlam)):
        tgt[i] = dlam[i - n_p - n_j]
    return tgt


def per_round_stall(t, eps, eps_max, sched, dt_s):
    """Total stall time overlapping each all-to-all phase (after startup), ms."""
    per_round = []
    for s, e, ph, _ in sched:
        if not ph.startswith("alltoall") or s < STARTUP_SKIP_S:
            continue
        i0, i1 = bisect.bisect_left(t, s), bisect.bisect_left(t, e)
        stall = sum(dt_s for i in range(i0, i1) if eps[i] > eps_max) * 1000.0
        per_round.append(stall)
    return per_round


def lorentz_rate_factor(eps_pm):
    return 1.0 / (1.0 + (2.0 * eps_pm / FWHM_PM) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--traces", default="L4seq4096")
    ap.add_argument("--sched-tag", default="L4seq4096",
                    help="suffix of the power-schedule CSVs (e.g. L4seq4096_hottest for per-device runs)")
    a = ap.parse_args()
    dt_s = a.dt_ms / 1000.0
    sched_tag = a.sched_tag
    print(f"FWHM={FWHM_PM:.1f} pm eps_max={EPS_MAX_PM:.1f} pm; traces={a.traces}; tracker grid {a.dt_ms} ms")
    results = {}
    for label, key, sched_fmt in MODELS:
        tpath = f"{key}_pic_transient_{a.traces}.csv"
        if not os.path.exists(tpath):
            print(f"[{label}] missing {tpath}, skipped")
            continue
        t_raw, T_raw = parse_prvar(tpath)
        t, T = resample(t_raw, T_raw, dt_s)
        sched = load_schedule(sched_fmt.format(tag=sched_tag))
        T0 = T[0]
        dlam = [(Ti - T0) * DLAM_DT_PM_PER_K for Ti in T]
        i_ss = bisect.bisect_left(t, STARTUP_SKIP_S)
        swing_pm = (max(T[i_ss:]) - min(T[i_ss:])) * DLAM_DT_PM_PER_K
        r_base = R_CTRL_BASE * DLAM_DT_PM_PER_K
        closed_form_ms = max(0.0, (swing_pm - EPS_MAX_PM) / r_base)

        eps = track(t, dlam, r_base)
        w = windows(t, eps, EPS_MAX_PM, STARTUP_SKIP_S)
        durs = [(e - s) * 1000 for s, e in w]
        rounds = per_round_stall(t, eps, EPS_MAX_PM, sched, dt_s)
        n_rounds = len(rounds)
        rate = [lorentz_rate_factor(e) for e in eps[i_ss:]]
        # rate loss only during communication phases
        at = phase_lookup(sched)
        comm_rate = [lorentz_rate_factor(eps[i]) for i in range(i_ss, len(t)) if at(t[i]).startswith("alltoall")]

        print(f"\n=== {label} ===")
        print(f"  steady swing {swing_pm:.0f} pm ({swing_pm/FWHM_PM:.1f} FWHM); closed-form stall {closed_form_ms:.0f} ms")
        print(f"  simulated (R_ctrl={R_CTRL_BASE} K/ms): {len(durs)} windows, mean {sum(durs)/len(durs):.1f} ms, max {max(durs):.1f} ms")
        print(f"  per a2a round ({n_rounds} rounds): stall overlapping comm mean {sum(rounds)/n_rounds:.1f} ms, "
              f"max {max(rounds):.1f} ms, rounds with any stall {sum(1 for r in rounds if r>0)}/{n_rounds}")
        print(f"  Lorentzian graded model: mean link-rate factor during comm {sum(comm_rate)/len(comm_rate):.3f} "
              f"(i.e. {100*(1-sum(comm_rate)/len(comm_rate)):.1f}% effective bandwidth loss), min {min(comm_rate):.3f}")

        # slew-rate sweep
        sweep = []
        for r_k in R_SWEEP_K_PER_MS:
            e2 = track(t, dlam, r_k * DLAM_DT_PM_PER_K)
            r2 = per_round_stall(t, e2, EPS_MAX_PM, sched, dt_s)
            sweep.append((r_k, sum(r2) / len(r2), max(r2)))
        thr = next((r for r, m, mx in sweep if mx == 0.0), None)
        print("  R_ctrl sweep (K/ms -> mean/max stall per round, ms): " +
              ", ".join(f"{r:g}: {m:.1f}/{mx:.1f}" for r, m, mx in sweep))
        print(f"  smallest swept R_ctrl with zero stall: {thr} K/ms "
              f"(closed form: swing/eps -> {swing_pm/EPS_MAX_PM:.0f}x the 'detune within one sample' rate)")

        # feed-forward controller: predict this iteration's detuning from the previous one
        ff_res = {}
        for err in (0.0, 0.05, 0.10, 0.20):
            ff = periodic_prediction_targets(t, dlam, sched, dt_s, model_error_frac=err)
            e_ff = track(t, dlam, r_base, feedforward=ff)
            r_ff = per_round_stall(t, e_ff, EPS_MAX_PM, sched, dt_s)
            ff_res[err] = (sum(r_ff) / len(r_ff), max(r_ff))
        print("  feed-forward (previous-iteration prediction) @ base R_ctrl, stall per round mean/max ms by "
              "prediction error: " + ", ".join(f"{int(e*100)}%: {m:.1f}/{mx:.1f}" for e, (m, mx) in ff_res.items()))
        jit_res = {}
        for jit in (2.0, 5.0, 10.0, 20.0):
            ff = jittered_prediction_targets(t, dlam, sched, dt_s, jit)
            e_ff = track(t, dlam, r_base, feedforward=ff)
            r_ff = per_round_stall(t, e_ff, EPS_MAX_PM, sched, dt_s)
            e_ff10 = track(t, dlam, r_base * 10, feedforward=ff)
            r_ff10 = per_round_stall(t, e_ff10, EPS_MAX_PM, sched, dt_s)
            jit_res[jit] = (sum(r_ff) / len(r_ff), max(r_ff), sum(r_ff10) / len(r_ff10))
        print("  feed-forward with timing jitter (ms) -> stall per round mean/max @ base R_ctrl | mean @ 10x: " +
              ", ".join(f"{j:g}: {m:.1f}/{mx:.1f} | {m10:.1f}" for j, (m, mx, m10) in jit_res.items()))
        r_ff = [ff_res[0.0][0]]
        r_ff_fast = [jit_res[10.0][0]]

        # heater bias / FSR
        swing_nm = swing_pm / 1000.0
        bias_mw = swing_nm / HEATER_EFF_NM_PER_MW
        print(f"  heater-only tuner: covering a {swing_nm:.2f} nm swing needs a >= {bias_mw:.1f} mW bias per ring "
              f"({bias_mw*RINGS_PER_DIE/1000:.1f} W per die for {RINGS_PER_DIE} rings) or an FSR hop (FSR {FSR_NM} nm) "
              f"whenever the resonance drifts past the bias point")

        results[label] = dict(swing_pm=swing_pm, closed_form_ms=closed_form_ms,
                              mean_window_ms=sum(durs) / len(durs), n_windows=len(durs),
                              per_round_mean_ms=sum(rounds) / n_rounds, per_round_max_ms=max(rounds),
                              comm_rate_factor=sum(comm_rate) / len(comm_rate),
                              r_sweep=sweep, r_zero_stall=thr,
                              ff_mean_ms=sum(r_ff) / len(r_ff), ff10x_mean_ms=sum(r_ff_fast) / len(r_ff_fast),
                              heater_bias_mw=bias_mw)
    with open(f"stall_model_v2_{a.traces}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote stall_model_v2_{a.traces}.json")


if __name__ == "__main__":
    main()
