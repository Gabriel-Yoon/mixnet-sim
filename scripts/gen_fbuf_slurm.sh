#!/bin/bash
#SBATCH --job-name=moe_taskgraph
#SBATCH --output=gen_fbuf_%j.txt
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#
# Generate MoE training task-graph .fbuf files with FlexFlow (mixnet-flexflow).
# Task-graph generation runs on a SINGLE GPU (-ll:gpu 1); the LOGICAL GPU count is
# set purely by the parallelism flags: num_gpus = train_dp * train_tp * train_pp.
# So 128 GPU = dp2 x tp8 x pp8, 256 GPU = dp4 x tp8 x pp8, etc. The .fbuf is what
# mixnet-htsim consumes (-flowfile) for packet-level fabric comparison.
#
# Edit FF_DIR and OUT_DIR for your HPC environment, then: sbatch gen_fbuf_slurm.sh
set -euo pipefail

FF_DIR="${FF_DIR:-$HOME/mixnet/FlexFlow-master}"     # built FlexFlow (CUDA) checkout
OUT_DIR="${OUT_DIR:-$HOME/mixnet/taskgraph}"          # where .fbuf files land
MOE="$FF_DIR/build/examples/cpp/mixture_of_experts/moe"
mkdir -p "$OUT_DIR"
cd "$FF_DIR"

# --- model presets (Mixtral-8x22B). Edit dims for other MoE models. ---
NUM_LAYERS=56 ; EMB=1024 ; EXP_HIDDEN=16384 ; HIDDEN=6144 ; HEADS=32 ; SEQ=4096 ; TOPK=2 ; EXPNUM=8
BATCH=128

# --- scale-out sweep: (label, dp, tp, pp) -> num_gpus = dp*tp*pp ---
#   keep num_gpus a multiple of 16 so it tiles cleanly into 4x4 glass panels.
declare -a CONFIGS=(
  "128gpu 2 8 8"     # 2*8*8 = 128  (8 panels)
  "256gpu 4 8 8"     # 4*8*8 = 256  (16 panels)
  "512gpu 8 8 8"     # 8*8*8 = 512  (32 panels)
)
declare -a MBS=(8 16 32)

for cfg in "${CONFIGS[@]}"; do
  read -r label dp tp pp <<< "$cfg"
  for mb in "${MBS[@]}"; do
    echo "=== gen $label mb$mb (dp$dp tp$tp pp$pp = $((dp*tp*pp)) GPU) ==="
    "$MOE" -ll:gpu 1 -ll:fsize 31000 -ll:zsize 24000 \
      --budget 20 --only-data-parallel --batchsize "$BATCH" --microbatchsize "$mb" \
      --train_dp "$dp" --train_tp "$tp" --train_pp "$pp" \
      --num-layers "$NUM_LAYERS" --embedding-size "$EMB" --expert-hidden-size "$EXP_HIDDEN" \
      --hidden-size "$HIDDEN" --num-heads "$HEADS" --sequence-length "$SEQ" \
      --topk "$TOPK" --expnum "$EXPNUM"
    mv taskgraph.fbuf "$OUT_DIR/mixtral8x22B_${label}_dp${dp}_tp${tp}_pp${pp}_ep${EXPNUM}_mb${mb}.fbuf"
    mv output.txt     "$OUT_DIR/mixtral8x22B_${label}_dp${dp}_tp${tp}_pp${pp}_ep${EXPNUM}_mb${mb}.txt" 2>/dev/null || true
  done
done
echo "ALL DONE -> $OUT_DIR"
