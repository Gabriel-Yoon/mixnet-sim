# Analytical MultiHeadAttention cost (FlexFlow moe fbuf generation)

## Why analytical (not measured)
FlexFlow's `measure_operator_cost` runs the op on the GPU via the **cuDNN multi-head
attention API** (`cudnnSetAttnDescriptor` + `cudnnMultiHeadAttnForward/Backward`). That
deprecated API is unstable for the large-model dims we need:
- `CUDNN_STATUS_BAD_PARAM` at `attention.cu:225` when `num_samples` (per-device batch) is 0
  — fixed by `--batch-size >= #devices`;
- then `CUDA_ERROR_MISALIGNED_ADDRESS` during the kernel run (reserveSpace ~671 MB) for
  qwen3/deepseek/kimi dims.
mixtral8x22B (smaller dims) measured fine, the big models do not. So for
`OP_MULTIHEAD_ATTENTION` we **replace the measurement with a FLOP-based analytical estimate**.
All other compute ops (the dominant expert-FFN `Dense`, layernorm, etc.) are still measured;
the MoE routing ops (`OP_GROUP_BY/OP_AGGREGATE/OP_TOPK`) are skipped (authors' own note — they
are not measurable and are negligible).

Implemented in `src/runtime/substitution.cc` at the **3** `measure_operator_cost` call sites
(≈ lines 1510, 1523, 2145).

## Notation (per device, per the op's own dims)
- `H` = `attn->qSize`        — hidden size (per device); for our configs 4096 (qwen3) / 7168 (deepseek,kimi)
- `S` = `attn->qoSeqLength`  — query/output sequence length per sample
- num_heads × head_dim = H   (so the per-head terms sum back to H)
- 1 sample per device — holds because the gen scripts set `--batch-size == #devices`
  (= dp·tp·pp·ep), so the data-parallel split gives `num_samples = 1` per device.

## FLOP count (forward, per device, 1 sample)
A matmul of `[m×k]·[k×n]` costs `2·m·k·n` FLOPs.

| term | what | FLOPs |
|------|------|-------|
| Q,K,V input projections | 3 × `[S×H]·[H×H]` | `3 · 2 · S · H² = 6·S·H²` |
| output projection        | 1 × `[S×H]·[H×H]` | `2·S·H²` |
| **projections subtotal** | | **`8·S·H²`** |
| scores `QKᵀ`            | per-head `[S×d]·[d×S]`, ×heads | `2·S²·H` |
| context `softmax(·)·V`  | per-head `[S×S]·[S×d]`, ×heads | `2·S²·H` |
| **attention subtotal**   | | **`4·S²·H`** |

```
fwd_flops = 8·S·H²  +  4·S²·H
bwd_flops = 2 · fwd_flops          # standard fwd:bwd ≈ 1:2 for GEMM-dominated ops
```
(Softmax/scale/bias elementwise work is O(S²·heads) ≪ the GEMMs and is ignored.)

## Time
```
forward_time_ms  = fwd_flops / (EFF_TFLOPS · 1e12) · 1e3
backward_time_ms = 2 · forward_time_ms
EFF_TFLOPS = 67           # = H200 fp32 peak; CALIBRATED to the measured Dense (see below)
```
`CostMetrics.forward_time/backward_time` are in **milliseconds** (CUDA-event units, matching the
measured ops; e.g. a measured expert `Dense` forward ≈ 2.2 ms in the same graph).

## Calibration of EFF_TFLOPS (against the measured Dense)
FlexFlow `measure_operator_cost` runs the real op in **fp32**. We back out the effective throughput
from a measured expert `Dense` GEMM and use the SAME value for the analytical attention (both are
GEMM-dominated, so they share hardware efficiency):
- a measured expert Dense (qwen3): fwd `2.198 ms`, mem in/out/weight = `5.03e7 / 1.34e8 / 5.04e7`
  bytes (fp32 → /4 = elements: `MK=1.26e7, MN=3.36e7, KN=1.26e7`).
- GEMM FLOPs `= 2·M·N·K = 2·√(MK·MN·KN) = 2·√(1.26e7·3.36e7·1.26e7) ≈ 1.46e11`.
- effective `= 1.46e11 / 2.198e-3 ≈ 66.3 TFLOPS ≈ H200 fp32 peak (67)`.
→ **EFF_TFLOPS = 67.** (The first draft used 200, which underestimated attention ~3× — 0.77 ms;
67 gives ~2.3 ms, matching the measured Dense scale.)

## Sanity check (qwen3, per device, H=4096, S=1024)
- `fwd_flops = 8·1024·4096² + 4·1024²·4096 ≈ 1.546e11`
- `forward_time = 1.546e11 / 67e12 · 1e3 ≈ 2.31 ms`, backward ≈ 4.62 ms

## Attention's weight in the makespan (critical path, NOT a sum)
A naive "sum of all op times" makes attention look ~0.1% — **misleading**, because the MoE has
128–384 **expert Dense ops that run in PARALLEL across EP devices**. The makespan (and htsim) use
the **critical path**, where each layer contributes **1 attention + 1 expert FFN** (not the sum of
all experts). On that path: attention ≈ 2.3 ms vs expert up+down ≈ 4.4 ms →
**attention ≈ 35–43% of per-layer compute**. So attention is significant, and getting EFF_TFLOPS
right matters; it is now calibrated to the measured Dense.

## Assumptions & limitations (state in the paper)
- 1 sample/device (gen scripts enforce `--batch-size == #devices`); for batch > devices,
  multiply `fwd_flops` by `num_samples`.
- Uses `qSize`/`qoSeqLength` as the op reports them (per-device frame, consistent with how the
  measured ops are scaled).
- `EFF_TFLOPS=200` is a single effective-throughput constant; attention is GEMM-dominated so a
  GEMM-efficiency constant is reasonable, but it is not a per-shape roofline.
- For the **fabric study** this only sets the compute that overlaps communication; the a2a/all-
  reduce volumes (the quantity under study) come from tensor sizes, unaffected. Reported results
  are normalized (ratios), so a uniform `EFF_TFLOPS` offset cancels across fabrics.
