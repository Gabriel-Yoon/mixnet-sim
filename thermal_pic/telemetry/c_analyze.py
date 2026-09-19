"""Job C analysis: per-pattern temperature swing, exponential time constants, MoE a2a timing.

usage: python c_analyze.py <synth_dir> <moe_dir>
Pure stdlib (the gemm-power env has no numpy).
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict

synth, moe = sys.argv[1], sys.argv[2]


def load_telemetry(path):
    per_gpu = defaultdict(list)  # idx -> [(t, temp, memtemp, power)]
    with open(path) as f:
        for r in csv.DictReader(f):
            per_gpu[int(r["nvml_index"])].append((
                float(r["wall_time_s"]), float(r["temp_gpu_C"]),
                float(r["temp_mem_C"]) if r["temp_mem_C"] else float("nan"),
                float(r["power_W"]),
            ))
    return per_gpu


def fit_tau(seg):
    """Least-squares fit T(t) = A + B*exp(-t/tau); grid over tau, linear solve for A,B."""
    if len(seg) < 20:
        return None
    t0 = seg[0][0]
    ts = [(t - t0) for t, _ in seg]
    ys = [y for _, y in seg]
    best = None
    tau = 0.2
    while tau <= 300:
        s1 = sum(math.exp(-t / tau) for t in ts)
        s2 = sum(math.exp(-2 * t / tau) for t in ts)
        sy = sum(ys)
        sey = sum(y * math.exp(-t / tau) for t, y in zip(ts, ys))
        n = len(ts)
        det = n * s2 - s1 * s1
        if abs(det) > 1e-12:
            A = (sy * s2 - s1 * sey) / det
            B = (n * sey - s1 * sy) / det
            ss = sum((y - (A + B * math.exp(-t / tau))) ** 2 for t, y in zip(ts, ys))
            if best is None or ss < best[0]:
                best = (ss, tau, A, B)
        tau *= 1.05
    ss, tau, A, B = best
    rmse = math.sqrt(ss / len(ts))
    return tau, A, B, rmse


print("=" * 70)
print("PART 2 — synthetic burst/idle patterns (1 GPU)")
tel = load_telemetry(f"{synth}/telemetry.csv")
idx = sorted(tel)[0]
samples = tel[idx]
print(f"telemetry: {len(samples)} samples on nvml{idx}, "
      f"median period {1e3 * st.median([b[0] - a[0] for a, b in zip(samples, samples[1:])]):.1f} ms, "
      f"span {samples[-1][0] - samples[0][0]:.1f} s")

phases = list(csv.DictReader(open(f"{synth}/phases.csv")))
by_pattern = defaultdict(list)
for p in phases:
    by_pattern[p["pattern"]].append(p)


def window(t0, t1):
    return [(t, temp) for t, temp, _, _ in samples if t0 <= t <= t1]


def power_window(t0, t1):
    return [p for t, _, _, p in samples if t0 <= t <= t1]


print(f"\n{'pattern':16} {'cycles':>6} {'T p2p (C)':>10} {'T mean':>7} {'T max':>6} "
      f"{'P idle':>7} {'P busy':>7} {'GEMMs/burst':>11}")
for name, rows in by_pattern.items():
    busy = [r for r in rows if r["phase"] == "busy"]
    if not busy:
        continue
    steady = rows[max(0, len(rows) - 20):]  # last ~10 cycles
    t0, t1 = float(steady[0]["start_s"]), float(steady[-1]["end_s"])
    w = window(t0, t1)
    temps = [x[1] for x in w]
    pidle = [p for r in rows if r["phase"] == "idle" for p in power_window(float(r["start_s"]), float(r["end_s"]))]
    pbusy = [p for r in busy for p in power_window(float(r["start_s"]), float(r["end_s"]))]
    print(f"{name:16} {len(busy):6d} {max(temps) - min(temps):10.1f} {st.mean(temps):7.1f} {max(temps):6.0f} "
          f"{(st.mean(pidle) if pidle else 0):7.1f} {(st.mean(pbusy) if pbusy else 0):7.1f} "
          f"{st.median([int(r['n_gemm']) for r in busy]):11.0f}")

print("\ntime-constant fits on the 60 s busy / 60 s idle pattern (T = A + B*exp(-t/tau)):")
d = by_pattern.get("d_60son_60soff", [])
for r in d:
    seg = window(float(r["start_s"]), float(r["end_s"]))
    if len(seg) < 50:
        continue
    t0 = seg[0][0]
    fit = fit_tau([(t, y) for t, y in seg])
    if not fit:
        continue
    tau, A, B, rmse = fit
    kind = "heating" if r["phase"] == "busy" else "cooling"
    print(f"  cycle {r['cycle']} {kind:8} T0={seg[0][1]:.0f}C -> T_inf={A:.1f}C "
          f"(delta {abs(B):.1f}C) tau={tau:.1f} s  rmse={rmse:.2f} C")

print("\n" + "=" * 70)
print("PART 3 — real MoE (8x H100, 4 layers, EP=8, top-2, seq 4096, bf16)")
iters = defaultdict(list)
for rank in range(8):
    try:
        for r in csv.DictReader(open(f"{moe}/iters_rank{rank}.csv")):
            iters[int(r["iter"])].append((float(r["iter_ms"]), float(r["a2a_ms"]),
                                          float(r["expert_gemm_ms"]), float(r["a2a_share"])))
    except FileNotFoundError:
        pass
warm = [i for i in sorted(iters) if i >= 2]
it_ms = [st.mean([x[0] for x in iters[i]]) for i in warm]
a2a_ms = [st.mean([x[1] for x in iters[i]]) for i in warm]
gemm_ms = [st.mean([x[2] for x in iters[i]]) for i in warm]
share = [st.mean([x[3] for x in iters[i]]) for i in warm]
print(f"iterations {warm[0]}..{warm[-1]} (rank-averaged):")
print(f"  iteration time   mean {st.mean(it_ms):8.1f} ms   median {st.median(it_ms):8.1f}   min {min(it_ms):8.1f}   max {max(it_ms):8.1f}")
print(f"  all-to-all total mean {st.mean(a2a_ms):8.1f} ms   median {st.median(a2a_ms):8.1f}   min {min(a2a_ms):8.1f}   max {max(a2a_ms):8.1f}")
print(f"  expert GEMM      mean {st.mean(gemm_ms):8.1f} ms   median {st.median(gemm_ms):8.1f}")
print(f"  a2a share of iteration: mean {100 * st.mean(share):.1f}%   min {100 * min(share):.1f}%   max {100 * max(share):.1f}%")

ev = defaultdict(list)
nbytes = {}
for rank in range(8):
    try:
        for r in csv.DictReader(open(f"{moe}/events_rank{rank}.csv")):
            if int(r["iter"]) < 2:
                continue
            key = (r["label"], r["direction"])
            ev[key].append(float(r["duration_ms"]))
            if r["kind"] == "a2a":
                nbytes[key] = int(r["nbytes"])
    except FileNotFoundError:
        pass
print(f"\n  {'event':22} {'n':>5} {'mean ms':>9} {'median':>8} {'p95':>8} {'max':>8} {'bytes/rank':>12}")
for key in sorted(ev):
    v = sorted(ev[key])
    p95 = v[int(0.95 * (len(v) - 1))]
    print(f"  {key[0] + ' ' + key[1]:22} {len(v):5d} {st.mean(v):9.3f} {st.median(v):8.3f} {p95:8.3f} {max(v):8.3f} "
          f"{nbytes.get(key, 0):12d}")

mt = load_telemetry(f"{moe}/telemetry.csv")
first_iter = min(float(r["start_s"]) for r in csv.DictReader(open(f"{moe}/iters_rank0.csv")))
last_iter = max(float(r["end_s"]) for r in csv.DictReader(open(f"{moe}/iters_rank0.csv")))
print(f"\n  per-GPU temperature during the {last_iter - first_iter:.0f} s training window:")
print(f"  {'gpu':>4} {'T start':>8} {'T max':>7} {'T p2p':>7} {'Tmem max':>9} {'P mean':>8} {'P max':>7}")
for g in sorted(mt):
    w = [(t, temp, mtp, p) for t, temp, mtp, p in mt[g] if first_iter <= t <= last_iter]
    if not w:
        continue
    temps = [x[1] for x in w]
    print(f"  {g:4d} {temps[0]:8.0f} {max(temps):7.0f} {max(temps) - min(temps):7.0f} "
          f"{max(x[2] for x in w):9.0f} {st.mean([x[3] for x in w]):8.1f} {max(x[3] for x in w):7.1f}")
