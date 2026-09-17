#!/usr/bin/env python3
"""Build a full forward+backward-pass GPU power schedule directly from the
SAME paper_L4 FlexFlow taskgraph that htsim uses for the Section 4.4 network
simulation (mixtral8x7B_paper_dp2tp4pp4_ep8top2_L4_seq1024_mb8_H100.fbuf /
llamaMoE_paper_dp2tp1pp4_ep16top2_L4_seq1024_mb8_H100.fbuf) -- replacing the
older, disconnected, forward-only ICCAD_2026 power schedule
(forward_time_breakdown_mb8.json) that previously drove the ANSYS transient.

Why this replaces the old schedule:
  - The old schedule's phase *durations* came from a different, older project
    with no traceable link to the taskgraph actually fed into htsim, and had
    no backward-pass phases at all (file literally named "forward10").
  - Confirmed empirically (live htsim run + log grep) that htsim DOES replay
    a full forward+backward iteration, including backward-direction
    GROUP_BY/AGGREGATE all-to-all events (32 fwd + 32 bwd each, symmetric).
    The thermal side should match that scope.

Per-device scaling: the taskgraph's node costs are the FULL exported
dataflow graph across all pipeline/expert shards, not one device's
timeline. dp2tp4pp4ep8 (mixtral) / dp2tp1pp4ep16 (llama-moe) map pp=4 to
"1 of the 4 layers per pipeline stage" and ep to "1 of N experts per
device" (the standard EP/PP sharding this project uses throughout).
So: attention/gate/addnorm (summed over all 4 layers) are divided by 4
(pp degree) to get one device's per-layer share; experts (summed over all
4 layers x N experts) are divided by (4 x N_experts) to get one device's
own single expert's share.

All-to-all (dispatch/combine) duration: the taskgraph's Group_by/Aggregate
nodes carry ZERO "secs" cost (they are TASK_ALLTOALL-typed; real duration
comes from htsim's own network replay, not a static op cost). Rather than
introduce a new, unverified bytes/bandwidth estimate, this keeps the
already-established alltoall duration from the prior (forward-only)
schedule (0.350002537 s per event) UNCHANGED and applies it symmetrically
to both the forward and the newly-added backward alltoall phases
(gradient tensors are the same size as activations, so a symmetric
duration is the natural minimal-new-assumption choice).
"""
import csv
import re
import sys

ALLTOALL_S = 0.350002537   # unchanged from the prior forward-only schedule
TDP_W = 700.0
PHASE_FRACTION = {
    "attention": 1.00,
    "gate": 0.40,
    "experts": 1.00,
    "addnorm": 0.20,
    "alltoall_1": 0.12,
    "alltoall_2": 0.12,
}
N_ITERS = 10


def parse_graph(path):
    txt = open(path).read()
    node_pat = re.compile(
        r'node(\d+) \[label="\{ ([A-Za-z0-9_ ]+?)_(\d+) \|.*?'
        r'\{ fwd \| bwd \| sync \| secs \} \| \{ ([0-9.eE+-]+) \| ([0-9.eE+-]+) \| [0-9.eE+-]+ \}'
    )
    nodes = {}
    for m in node_pat.finditer(txt):
        nid, opname, opid, fwd, bwd = m.groups()
        nodes[int(nid)] = dict(opname=opname.strip(), opid=int(opid), fwd=float(fwd), bwd=float(bwd))
    edge_pat = re.compile(r'node(\d+) -> node(\d+);')
    edges = [(int(a), int(b)) for a, b in edge_pat.findall(txt)]
    return nodes, edges


def classify(nodes, edges):
    succs = {}
    for a, b in edges:
        succs.setdefault(a, []).append(b)

    gb_ids = [nid for nid, n in nodes.items() if n["opname"] == "Group_by"]
    mha_ids = [nid for nid, n in nodes.items() if n["opname"] == "MultiHeadAttention"]

    expert_nodes = set()
    for g in gb_ids:
        stack, seen = list(succs.get(g, [])), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if nodes[cur]["opname"] == "Aggregate cooperation":
                continue
            expert_nodes.add(cur)
            stack.extend(succs.get(cur, []))

    attn_residual_add = set()
    for m in mha_ids:
        for s in succs.get(m, []):
            if nodes[s]["opname"] == "Add":
                attn_residual_add.add(s)

    phase = {}
    for nid, n in nodes.items():
        op = n["opname"]
        if op in ("Input", "Repartition", "MultiHeadAttention"):
            phase[nid] = "attention"
        elif op == "Add" and nid in attn_residual_add:
            phase[nid] = "attention"
        elif op == "LayerNorm" or (op == "Add" and nid not in attn_residual_add):
            phase[nid] = "addnorm"
        elif op == "Group_by":
            phase[nid] = "alltoall_1"
        elif op == "Aggregate cooperation":
            phase[nid] = "alltoall_2"
        elif op == "TopK":
            phase[nid] = "gate"
        elif op in ("Dense", "Softmax"):
            phase[nid] = "experts" if nid in expert_nodes else "gate"
        else:
            phase[nid] = "UNKNOWN:" + op
    return phase


def phase_sums(taskgraph_path, n_layers, n_experts):
    nodes, edges = parse_graph(taskgraph_path)
    phase = classify(nodes, edges)
    sums = {k: [0.0, 0.0] for k in PHASE_FRACTION}
    for nid, ph in phase.items():
        if ph not in sums:
            continue
        sums[ph][0] += nodes[nid]["fwd"]
        sums[ph][1] += nodes[nid]["bwd"]
    # per-device scaling: pp degree (n_layers) for attention/gate/addnorm,
    # + expert-parallel degree (n_experts) additionally for experts.
    out = {}
    for ph in ("attention", "gate", "addnorm"):
        f, b = sums[ph]
        out[ph] = (f / n_layers, b / n_layers)
    f, b = sums["experts"]
    out["experts"] = (f / (n_layers * n_experts), b / (n_layers * n_experts))
    out["alltoall_1"] = (ALLTOALL_S * 1000.0, ALLTOALL_S * 1000.0)  # ms, fwd==bwd
    out["alltoall_2"] = (ALLTOALL_S * 1000.0, ALLTOALL_S * 1000.0)
    return out


def build_schedule(model_label, durations_ms, n_iters=N_ITERS):
    fwd_order = ["attention", "gate", "alltoall_1", "experts", "alltoall_2", "addnorm"]
    bwd_order = list(reversed(fwd_order))
    rows = []
    t = 0.0
    for it in range(1, n_iters + 1):
        for ph in fwd_order:
            dur_s = durations_ms[ph][0] / 1000.0
            rows.append((model_label, it, "fwd", ph, t, t + dur_s, dur_s, TDP_W * PHASE_FRACTION[ph]))
            t += dur_s
        for ph in bwd_order:
            dur_s = durations_ms[ph][1] / 1000.0
            rows.append((model_label, it, "bwd", ph, t, t + dur_s, dur_s, TDP_W * PHASE_FRACTION[ph]))
            t += dur_s
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "iteration", "direction", "phase", "start_s", "end_s", "duration_s", "power_w"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.9f}", f"{r[5]:.9f}", f"{r[6]:.9f}", f"{r[7]:.4f}"])
    print(f"wrote {out_path}  ({len(rows)} rows, total {rows[-1][5]:.3f} s)")


MODELS = {
    "mixtral8x7b": dict(
        path="taskgraph/mixtral8x7B_paper_dp2tp4pp4_ep8top2_L4_seq1024_mb8_H100.txt",
        n_layers=4, n_experts=8,
    ),
    "llama-moe-6.7b": dict(
        path="taskgraph/llamaMoE_paper_dp2tp1pp4_ep16top2_L4_seq1024_mb8_H100.txt",
        n_layers=4, n_experts=16,
    ),
}

if __name__ == "__main__":
    for label, cfg in MODELS.items():
        durations_ms = phase_sums(cfg["path"], cfg["n_layers"], cfg["n_experts"])
        print(f"=== {label} === per-device per-iteration phase (fwd_ms, bwd_ms):")
        for ph, (f, b) in durations_ms.items():
            print(f"  {ph:12s} fwd={f:9.4f} ms  bwd={b:9.4f} ms")
        rows = build_schedule(label, durations_ms)
        out = f"{label.replace('-', '_')}_power_schedule_fwdbwd.csv"
        write_csv(rows, out)
