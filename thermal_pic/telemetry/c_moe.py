"""Job C part 3: 4-layer Mixtral-style MoE (H=4096, 8 experts, top-2, EP=8) forward+backward on 8 GPUs.

torchrun --nproc_per_node 8 c_moe.py OUT
Each rank owns one expert (SwiGLU 4096->14336->4096). Tokens are dispatched and
combined with dist.all_to_all_single inside a custom autograd Function so the
backward all-to-alls are timed too. CUDA events bracket every all-to-all and
every expert GEMM block (forward and backward); per-iteration anchors map event
offsets onto wall time. Rank 0 runs the NVML sampler for all GPUs.
Outputs: OUT/events_rank<r>.csv, OUT/iters_rank<r>.csv, OUT/telemetry.csv.
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

H, NE, TOPK, NL, SEQ, NSEQ, FFN, HEADS = 4096, 8, 2, 4, 4096, 8, 14336, 32
ITERS = 20
DT = torch.bfloat16

OUT = sys.argv[1]
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == NE, "one expert per rank (EP=8)"
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)

EVENTS = []  # (label, layer, direction, kind, start_event, end_event, nbytes)


def record(label, layer, direction, kind, nbytes=0):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    EVENTS.append([label, layer, direction, kind, s, e, nbytes])
    return s, e


class A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, layer, label):
        ctx.send_splits, ctx.recv_splits, ctx.layer, ctx.label = send_splits, recv_splits, layer, label
        out = x.new_empty((sum(recv_splits),) + tuple(x.shape[1:]))
        s, e = record(label, layer, "fwd", "a2a", x.numel() * x.element_size())
        s.record()
        dist.all_to_all_single(out, x.contiguous(), recv_splits, send_splits)
        e.record()
        return out

    @staticmethod
    def backward(ctx, g):
        out = g.new_empty((sum(ctx.send_splits),) + tuple(g.shape[1:]))
        s, e = record(ctx.label, ctx.layer, "bwd", "a2a", g.numel() * g.element_size())
        s.record()
        dist.all_to_all_single(out, g.contiguous(), ctx.send_splits, ctx.recv_splits)
        e.record()
        return out, None, None, None, None


class Mark(torch.autograd.Function):
    """Identity that records a CUDA event in forward and in backward."""

    @staticmethod
    def forward(ctx, x, ev_fwd, ev_bwd):
        ctx.ev_bwd = ev_bwd
        ev_fwd.record()
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        ctx.ev_bwd.record()
        return g, None, None


def expert_block(x, layer, w1, w2, w3):
    fs, fe = record("expert", layer, "fwd", "gemm")
    bs, be = record("expert", layer, "bwd", "gemm")
    x = Mark.apply(x, fs, be)  # fwd start / bwd end
    y = F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)
    return Mark.apply(y, fe, bs)  # fwd end / bwd start


def P(*shape, std=0.02):
    return (torch.randn(*shape, device=dev, dtype=torch.float32) * std).to(DT).requires_grad_()


torch.manual_seed(1234)  # same attention/router weights on every rank
layers = []
for _ in range(NL):
    layers.append(dict(
        qkv=P(3 * H, H), o=P(H, H), router=P(NE, H),
        n1=torch.ones(H, device=dev, dtype=DT, requires_grad=True),
        n2=torch.ones(H, device=dev, dtype=DT, requires_grad=True),
    ))
torch.manual_seed(1234 + rank)  # distinct local expert weights
experts = [dict(w1=P(FFN, H), w2=P(H, FFN), w3=P(FFN, H)) for _ in range(NL)]


def moe(x, li, L, E):
    T = x.shape[0]
    logits = F.linear(x, L["router"]).float()
    wts, idx = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    wts = (wts / wts.sum(-1, keepdim=True)).to(DT)
    flat_idx = idx.reshape(-1)
    order = torch.argsort(flat_idx, stable=True)
    tokens = x.repeat_interleave(TOPK, dim=0)[order]
    send = torch.bincount(flat_idx, minlength=NE)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    send_l, recv_l = send.tolist(), recv.tolist()
    h = A2A.apply(tokens, send_l, recv_l, li, "dispatch")
    h = expert_block(h, li, E["w1"], E["w2"], E["w3"])
    h = A2A.apply(h, recv_l, send_l, li, "combine")
    out = torch.empty_like(h)
    out[order] = h
    out = out.view(T, TOPK, H) * wts.unsqueeze(-1)
    return out.sum(1)


def layer_fwd(x, li):
    L, E = layers[li], experts[li]
    b = x.shape[0]
    h = F.rms_norm(x, (H,), L["n1"])
    q, k, v = F.linear(h, L["qkv"]).view(b, SEQ, 3, HEADS, H // HEADS).unbind(2)
    a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    x = x + F.linear(a.transpose(1, 2).reshape(b, SEQ, H), L["o"])
    h = F.rms_norm(x, (H,), L["n2"]).reshape(-1, H)
    return x + moe(h, li, L, E).view(b, SEQ, H)


sampler = None
if rank == 0:
    from c_telemetry import Sampler
    sampler = Sampler(os.path.join(OUT, "telemetry.csv"), period_s=0.02)
    sampler.start()
    time.sleep(30)  # idle baseline before the first iteration

ev_f = open(os.path.join(OUT, f"events_rank{rank}.csv"), "w")
ev_f.write("iter,label,layer,direction,kind,start_s,end_s,duration_ms,nbytes\n")
it_f = open(os.path.join(OUT, f"iters_rank{rank}.csv"), "w")
it_f.write("iter,start_s,end_s,iter_ms,a2a_ms,expert_gemm_ms,a2a_share\n")

for it in range(ITERS):
    EVENTS.clear()
    torch.manual_seed(10_000 + it * 100 + rank)
    x = torch.randn(NSEQ, SEQ, H, device=dev, dtype=DT)
    torch.cuda.synchronize()
    anchor = torch.cuda.Event(enable_timing=True)
    t_start = time.time()
    anchor.record()
    y = x
    for li in range(NL):
        y = layer_fwd(y, li)
    y.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    t_end = time.time()
    a2a = gemm = 0.0
    for label, layer, direction, kind, s, e, nb in EVENTS:
        st = t_start + anchor.elapsed_time(s) / 1e3
        d = s.elapsed_time(e)
        a2a += d if kind == "a2a" else 0.0
        gemm += d if kind == "gemm" else 0.0
        ev_f.write(f"{it},{label},{layer},{direction},{kind},{st:.6f},{st + d / 1e3:.6f},{d:.4f},{nb}\n")
    ims = (t_end - t_start) * 1e3
    it_f.write(f"{it},{t_start:.6f},{t_end:.6f},{ims:.3f},{a2a:.3f},{gemm:.3f},{a2a / ims:.4f}\n")
    for p in [v for L in layers for v in L.values()] + [v for E in experts for v in E.values()]:
        p.grad = None
    if rank == 0:
        print(f"iter {it}: {ims:.1f} ms, a2a {a2a:.1f} ms ({100 * a2a / ims:.1f}%), expert GEMM {gemm:.1f} ms", flush=True)

ev_f.close()
it_f.close()
if rank == 0:
    time.sleep(60)  # cool-down tail
    sampler.stop()
dist.destroy_process_group()
