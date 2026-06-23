#!/bin/bash
#SBATCH --job-name=moe_taskgraph
#SBATCH --output=gen_fbuf_%j.txt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#
# Generate MoE training task-graph .fbuf files with FlexFlow (mixnet-flexflow) for
# the paper's target models (DeepSeek-V3, Kimi-K2, Qwen3-235B) + Mixtral reference.
#
# Task-graph generation runs on a SINGLE GPU (-ll:gpu 1). The LOGICAL GPU count is
# set by the parallelism flags: num_gpus = train_dp * train_tp * train_pp. The .fbuf
# is consumed by mixnet-htsim (-flowfile) for packet-level fabric comparison.
#
# *** WHY THESE MODELS / THIS PARALLELISM ***
# The fabric story (glass-FB vs Fat-tree / MixNet / NVL72) needs an a2a-DOMINANT
# regime. a2a volume grows with #experts*topk; all-reduce volume grows with TP. So
# we use the real high-expert configs (256/384/128 experts, topk 8) with LOW TP and
# HIGH effective EP. A TP-heavy config (e.g. mixtral tp8) is all-reduce-dominant and
# hides the fabric effect (and starves MixNet's a2a reconfiguration). Keep TP small.
#
# *** PRACTICAL: REDUCE LAYERS ***
# Full depth (61/94 layers) makes both .fbuf gen and htsim very slow, and the per-layer
# comm pattern just repeats. Use a few representative layers (NLAYERS below) -- the
# fabric makespan RATIO is layer-count-invariant. Bump up only for a final headline run.
#
# Edit FF_DIR / OUT_DIR, then: sbatch gen_fbuf_slurm.sh
set -euo pipefail

FF_DIR="${FF_DIR:-$HOME/mixnet/FlexFlow-master}"     # built FlexFlow (CUDA) checkout
OUT_DIR="${OUT_DIR:-$HOME/mixnet/taskgraph}"          # where .fbuf files land
MOE="$FF_DIR/build/examples/cpp/mixture_of_experts/moe"
mkdir -p "$OUT_DIR"
cd "$FF_DIR"

NLAYERS="${NLAYERS:-4}"        # representative layers (raise for final run; ratio is invariant)
SEQ=4096 ; TOPK=8 ; BATCH=128
declare -a MBS=(8 16 32)

# --- target models: label | hidden | heads | expert_hidden(moe_intermediate) | experts ---
declare -a MODELS=(
  "deepseekV3 7168 128 2048 256"
  "kimiK2     7168  64 2048 384"
  "qwen3_235B 4096  64 1536 128"
)

# --- parallelism (num_gpus = dp*tp*pp). LOW TP for a2a-dominance. EP = expnum (per FlexFlow). ---
#   tune dp/pp to hit target GPU counts; keep tp small (1-2). MLA models (DeepSeek/Kimi) need little TP.
declare -a CONFIGS=(
  "128gpu 8 2 8"    # dp8  tp2 pp8 = 128
  "256gpu 16 2 8"   # dp16 tp2 pp8 = 256
)

for m in "${MODELS[@]}"; do
  read -r label hidden heads ehid experts <<< "$m"
  for cfg in "${CONFIGS[@]}"; do
    read -r glabel dp tp pp <<< "$cfg"
    for mb in "${MBS[@]}"; do
      echo "=== $label $glabel mb$mb (dp$dp tp$tp pp$pp = $((dp*tp*pp)) GPU, ${experts}e/top$TOPK, ${NLAYERS}L) ==="
      "$MOE" -ll:gpu 1 -ll:fsize 31000 -ll:zsize 24000 \
        --budget 20 --only-data-parallel --batchsize "$BATCH" --microbatchsize "$mb" \
        --train_dp "$dp" --train_tp "$tp" --train_pp "$pp" \
        --num-layers "$NLAYERS" --embedding-size "$hidden" --expert-hidden-size "$ehid" \
        --hidden-size "$hidden" --num-heads "$heads" --sequence-length "$SEQ" \
        --topk "$TOPK" --expnum "$experts"
      out="${label}_${glabel}_dp${dp}_tp${tp}_pp${pp}_ep${experts}_mb${mb}_${NLAYERS}L"
      mv taskgraph.fbuf "$OUT_DIR/${out}.fbuf"
      mv output.txt     "$OUT_DIR/${out}.txt" 2>/dev/null || true
    done
  done
done
echo "ALL DONE -> $OUT_DIR"
