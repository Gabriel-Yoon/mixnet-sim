#!/usr/bin/env python3
"""Build a per-device GPU power schedule from an htsim wafer log (COMPUTE_TASK + A2A_ROUND
lines), replacing the synthetic schedule of gen_power_schedule_fwdbwd.py.

For the chosen device, every compute task becomes a phase at the power of its class
(attention/expert GEMM 700 W, gate 280 W, add-norm 140 W); every gap between compute tasks
(all-to-all waits, pipeline bubbles) is idle at 84 W. The timeline covers the whole simulated
iteration (all micro-batches, all layers on that device) and is repeated N_ITERS times so the
ANSYS transient reaches a periodic steady state.

Task classification: FlexFlow op names embedded in the htsim task name. If a taskgraph .txt is
given, expert-vs-gate Dense nodes are resolved with gen_power_schedule_fwdbwd.classify();
otherwise a Dense/Softmax task shorter than GATE_MAX_MS is treated as gate.

  python3 gen_power_schedule_from_htsim.py <log[.gz]> <device_id|hot|median|cold> <out.csv>
      [--taskgraph <fbuf .txt>] [--iters 10] [--model label]
"""
import argparse
import csv
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_power_schedule_fwdbwd import TDP_W, PHASE_FRACTION, parse_graph, classify  # noqa: E402

IDLE_W = TDP_W * PHASE_FRACTION["alltoall_1"]   # 84 W
GATE_MAX_MS = 1.5
PAT = re.compile(r'COMPUTE_TASK dev=(-?\d+) id=(\d+) name="([^"]*)" type=(\w+) layer=(-?\d+) '
                 r'mb=(-?\d+) start_ps=(\d+) finish_ps=(\d+)')
PAT_A2A = re.compile(r'A2A_ROUND id=(\d+) layer=(-?\d+) mb=(-?\d+) info="([^"]*)" '
                     r'ready_ps=(\d+) start_ps=(\d+) finish_ps=(\d+)')


def read_a2a_rounds(path):
    op = gzip.open if path.endswith(".gz") else open
    rounds = []
    with op(path, "rt", errors="replace") as f:
        for line in f:
            m = PAT_A2A.search(line)
            if m:
                rid, layer, mb, info, ready, start, fin = m.groups()
                rounds.append(dict(id=int(rid), layer=int(layer), mb=int(mb), info=info,
                                   ready=int(ready) / 1e12, start=int(start) / 1e12, finish=int(fin) / 1e12))
    return rounds


def read_tasks(path):
    op = gzip.open if path.endswith(".gz") else open
    tasks = []
    with op(path, "rt", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                dev, tid, name, typ, layer, mb, s, e = m.groups()
                tasks.append(dict(dev=int(dev), id=int(tid), name=name, type=typ, layer=int(layer),
                                  mb=int(mb), start=int(s) / 1e12, end=int(e) / 1e12))
    return tasks


def phase_of(task, graph_phase):
    name = task["name"]
    m = re.match(r"([A-Za-z_ ]+?)_(\d+)", name)
    op = m.group(1).strip() if m else name
    opid = int(m.group(2)) if m else -1
    if graph_phase is not None and (op, opid) in graph_phase:
        return graph_phase[(op, opid)]
    if op in ("MultiHeadAttention", "Input", "Repartition"):
        return "attention"
    if op in ("LayerNorm",):
        return "addnorm"
    if op == "Add":
        return "addnorm"
    if op == "TopK":
        return "gate"
    if op in ("Dense", "Softmax"):
        return "gate" if (task["end"] - task["start"]) * 1000 < GATE_MAX_MS else "experts"
    return "experts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("device")
    ap.add_argument("out")
    ap.add_argument("--taskgraph", default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--model", default="htsim")
    a = ap.parse_args()

    tasks = read_tasks(a.log)
    if not tasks:
        sys.exit("no COMPUTE_TASK lines found")
    graph_phase = None
    if a.taskgraph:
        nodes, edges = parse_graph(a.taskgraph)
        ph = classify(nodes, edges)
        graph_phase = {(nodes[n]["opname"], nodes[n]["opid"]): ph[n] for n in nodes}

    busy = {}
    for t in tasks:
        busy[t["dev"]] = busy.get(t["dev"], 0.0) + (t["end"] - t["start"])
    devs = sorted(busy, key=lambda d: busy[d])
    if a.device in ("hot", "median", "cold"):
        dev = {"cold": devs[0], "median": devs[len(devs) // 2], "hot": devs[-1]}[a.device]
    else:
        dev = int(a.device)
    mine = sorted([t for t in tasks if t["dev"] == dev], key=lambda t: t["start"])
    t_end = max(t["end"] for t in tasks)          # iteration length = last compute finish
    print(f"devices={len(devs)} chosen dev={dev} busy={busy[dev]*1000:.1f} ms of {t_end*1000:.1f} ms "
          f"({busy[dev]/t_end*100:.0f}% duty); busiest dev {devs[-1]} {busy[devs[-1]]/t_end*100:.0f}%, "
          f"least {devs[0]} {busy[devs[0]]/t_end*100:.0f}%")

    # one iteration's phase list for this device: compute tasks (merged if contiguous & same
    # phase) with idle gaps in between
    phases = []
    cur = 0.0
    for t in mine:
        if t["start"] > cur + 1e-9:
            phases.append(("idle", cur, t["start"], IDLE_W))
        ph = phase_of(t, graph_phase)
        pw = TDP_W * PHASE_FRACTION["experts" if ph == "experts" else ph]
        s = max(t["start"], cur)
        if phases and phases[-1][0] == ph and abs(phases[-1][2] - s) < 1e-9:
            phases[-1] = (ph, phases[-1][1], t["end"], pw)
        else:
            phases.append((ph, s, t["end"], pw))
        cur = max(cur, t["end"])
    if cur < t_end:
        phases.append(("idle", cur, t_end, IDLE_W))

    counts = {}
    for ph, s, e, _ in phases:
        counts[ph] = counts.get(ph, 0.0) + (e - s)
    print("  per-iteration time by phase (ms): " + ", ".join(f"{k}={v*1000:.1f}" for k, v in counts.items()))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "iteration", "direction", "phase", "start_s", "end_s", "duration_s", "power_w"])
        for it in range(1, a.iters + 1):
            off = (it - 1) * t_end
            for ph, s, e, pw in phases:
                if e - s < 1e-7:
                    continue
                w.writerow([f"{a.model}-dev{dev}", it, "sim", ph, f"{off+s:.9f}", f"{off+e:.9f}", f"{e-s:.9f}", f"{pw:.4f}"])
    print(f"wrote {a.out}  ({len(phases)} phases per iteration x {a.iters} iterations, total {t_end*a.iters:.3f} s)")

    # all-to-all rounds this device takes part in (rounds of the layer it hosts), repeated per
    # iteration like the schedule; used by stall_model_v2 --a2a-windows for per-round stall
    layers = sorted({t["layer"] for t in mine if t["layer"] >= 0})
    rounds = [r for r in read_a2a_rounds(a.log) if r["layer"] in layers]
    win_out = a.out.replace(".csv", "_a2a_windows.csv")
    with open(win_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "round_id", "layer", "mb", "info", "ready_s", "start_s", "finish_s"])
        for it in range(1, a.iters + 1):
            off = (it - 1) * t_end
            for r in sorted(rounds, key=lambda r: r["ready"]):
                w.writerow([it, r["id"], r["layer"], r["mb"], r["info"], f"{off+r['ready']:.9f}",
                            f"{off+r['start']:.9f}", f"{off+r['finish']:.9f}"])
    print(f"wrote {win_out}  (device layer(s) {layers}: {len(rounds)} a2a rounds per iteration)")


if __name__ == "__main__":
    main()
