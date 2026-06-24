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
EFF_TFLOPS = 200          # effective GEMM throughput on H100/H200 (TF32/fp32 regime), TUNABLE
```
`CostMetrics.forward_time/backward_time` are in **milliseconds** (CUDA-event units, matching the
measured ops; e.g. a measured `Dense` forward ≈ 2.2 ms in the same graph).

## Calibration / sanity check
qwen3 per device (H=4096, S=1024, 1 sample):
- `fwd_flops = 8·1024·4096² + 4·1024²·4096 = 1.37e11 + 1.72e10 ≈ 1.54e11`
- `forward_time = 1.54e11 / 2.0e14 · 1e3 ≈ 0.77 ms`, backward ≈ 1.54 ms
→ same order as the measured `Dense` (≈2.2 ms) in the graph → realistic proportion (not 0,
not absurd). If the measured-vs-analytical proportion looks off after a run, tune `EFF_TFLOPS`.

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
