#!/usr/bin/env python3
"""Extract per-all-to-all-round durations from an htsim wafer log.

Requires the `A2A_ROUND ...` lines emitted by FFTask::cleanup() (mixnet-htsim
wafer-topology branch). Outputs one CSV row per round plus a summary:
  - duration_ms   = finish - start   (pure network time of the round)
  - wait_ms       = start  - ready   (queueing behind the device / thermal stall)
  - number of rounds, mean/median/max duration, and the number of rounds on the
    serialized critical path (rounds whose ready time >= previous round's finish).

  python3 parse_a2a_rounds.py <log> <model> > a2a_rounds_<model>.csv
"""
import re
import statistics
import sys

pat = re.compile(
    r'A2A_ROUND id=(\d+) layer=(-?\d+) mb=(-?\d+) info="([^"]*)" '
    r'ready_ps=(\d+) start_ps=(\d+) finish_ps=(\d+)')

log, model = sys.argv[1], sys.argv[2]
rows = []
with open(log, errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if m:
            tid, layer, mb, info, ready, start, fin = m.groups()
            rows.append((int(tid), int(layer), int(mb), info, int(ready), int(start), int(fin)))

rows.sort(key=lambda r: r[6])
print("model,task_id,layer,mb,info,ready_ms,start_ms,finish_ms,wait_ms,duration_ms")
for tid, layer, mb, info, ready, start, fin in rows:
    print(f"{model},{tid},{layer},{mb},{info},{ready/1e9:.4f},{start/1e9:.4f},{fin/1e9:.4f},"
          f"{(start-ready)/1e9:.4f},{(fin-start)/1e9:.4f}")

dur = [(r[6] - r[5]) / 1e9 for r in rows]
chain, last_fin = 0, -1
for r in rows:
    if r[4] >= last_fin:
        chain += 1
        last_fin = r[6]
    else:
        last_fin = max(last_fin, r[6])
fwd = [d for r, d in zip(rows, dur) if "forward" in r[3].lower() or "fwd" in r[3].lower()]
bwd = [d for r, d in zip(rows, dur) if "backward" in r[3].lower() or "bwd" in r[3].lower()]
summ = (f"# {model}: rounds={len(rows)} mean={statistics.mean(dur):.3f} ms "
        f"median={statistics.median(dur):.3f} ms max={max(dur):.3f} ms "
        f"fwd_mean={statistics.mean(fwd) if fwd else float('nan'):.3f} "
        f"bwd_mean={statistics.mean(bwd) if bwd else float('nan'):.3f} "
        f"serialized_chain_rounds={chain} makespan_ms={max(r[6] for r in rows)/1e9:.3f}")
print(summ)
print(summ, file=sys.stderr)
