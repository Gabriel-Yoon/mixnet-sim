#!/bin/bash
# Generate MoE training task-graph .fbuf for the paper's target models, using the
# PROVEN HPC invocation (apptainer flexflow-cuda-11.8.sif). Run this INSIDE the
# apptainer shell after building FlexFlow:
#
#   module load cuda/11.7.0 cudnn/8.5.0.96-11.7-cuda
#   salloc -A gts-syu334-ece -qinferno -N1 --mem-per-gpu=32G -t2:00:00 --gres=gpu:V100:1
#   apptainer exec --cleanenv --nv /storage/project/r-syu334-0/shared/images/flexflow-cuda-11.8.sif bash
#   cmake .. -DFF_GPU_BACKEND=cuda -DFF_CUDA_ARCH=80 -DCMAKE_CUDA_ARCHITECTURES=80 && make -j8
#   bash <this script>
#
# Proven-working baseline (mixtral8x22B, TP8 -> all-reduce-heavy):
#   --train_dp 2 --train_tp 8 --train_pp 1 --num-layers 1 --embedding-size 1024 \
#   --expert-hidden-size 16384 --hidden-size 6144 --num-heads 32 --sequence-length 1024 \
#   --topk 2 --expnum 8 --batch-size 128 --simulator-workspace-size 12000000000
#
# For the fabric study we want an a2a-DOMINANT regime: real high-expert configs
# (256/384/128 experts, topk 8) with LOW TP (TP drives all-reduce, which hides the
# fabric effect and starves MixNet's a2a reconfiguration). So: --train_tp 1.
set -euo pipefail

MOE="${MOE:-./examples/cpp/mixture_of_experts/moe}"   # path to built moe binary
OUT="${OUT:-./results}"; mkdir -p "$OUT"
NLAYERS="${NLAYERS:-1}"   # 1 proven; raise for final headline (ratio is layer-invariant)
SEQ="${SEQ:-4096}"        # proven baseline used 1024; 4096 is realistic. start 1024 if OOM.
BATCH=128
declare -a MBS=(8 16 32)

# label | hidden | heads | expert_hidden(moe_intermediate) | experts | topk
declare -a MODELS=(
  "deepseekV3 7168 128 2048 256 8"
  "kimiK2     7168  64 2048 384 8"
  "qwen3_235B 4096  64 1536 128 8"
)
# label | dp tp pp  (num_gpus = dp*tp*pp; LOW tp for a2a-dominance)
declare -a CONFIGS=(
  "128gpu 16 1 8"   # 16*1*8 = 128
  "256gpu 32 1 8"   # 32*1*8 = 256
)

for m in "${MODELS[@]}"; do
  read -r label hidden heads ehid experts topk <<< "$m"
  for cfg in "${CONFIGS[@]}"; do
    read -r glabel dp tp pp <<< "$cfg"
    for mb in "${MBS[@]}"; do
      echo "=== $label $glabel mb$mb (dp$dp tp$tp pp$pp=$((dp*tp*pp))GPU, ${experts}e/top$topk, ${NLAYERS}L, seq$SEQ) ==="
      # NOTE: large expnum (256/384) may need bigger fsize/zsize/simulator-workspace; bump if OOM.
      "$MOE" -ll:gpu 1 -ll:fsize 40000 -ll:zsize 24000 \
        --budget 20 --only-data-parallel --simulator-workspace-size 12000000000 \
        --batch-size "$BATCH" --microbatchsize "$mb" \
        --train_dp "$dp" --train_tp "$tp" --train_pp "$pp" \
        --num-layers "$NLAYERS" --embedding-size "$hidden" --expert-hidden-size "$ehid" \
        --hidden-size "$hidden" --num-heads "$heads" --sequence-length "$SEQ" \
        --topk "$topk" --expnum "$experts"
      out="${label}_${glabel}_dp${dp}_tp${tp}_pp${pp}_ep${experts}_mb${mb}_${NLAYERS}L"
      mv taskgraph.fbuf "$OUT/${out}.fbuf"
      mv output.txt     "$OUT/${out}.txt" 2>/dev/null || true
    done
  done
done
echo "ALL DONE -> $OUT"
