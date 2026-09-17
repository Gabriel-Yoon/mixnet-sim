#!/bin/bash
#SBATCH -J pic_transient_scaling
#SBATCH -A gts-syu334-ece
#SBATCH -q inferno
#SBATCH -N1 -n8
#SBATCH --mem=16G
#SBATCH -t4:00:00
#SBATCH -o /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_scaling.out
#SBATCH -e /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_scaling.err
cd /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic
module load ansys/2025R2
rm -f *.lock
date
ansys252 -b -j pic_mixtral_L4seq4096 -np 8 -smp -i pic_transient_mixtral8x7b_L4seq4096.inp -o _out_mixtral_L4seq4096.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_mixtral_L4seq4096.out && echo 'mixtral L4seq4096 OK' || echo 'mixtral L4seq4096 HAD ERRORS'
ansys252 -b -j pic_llama_L4seq4096 -np 8 -smp -i pic_transient_llamamoe_L4seq4096.inp -o _out_llama_L4seq4096.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_llama_L4seq4096.out && echo 'llama L4seq4096 OK' || echo 'llama L4seq4096 HAD ERRORS'
ansys252 -b -j pic_mixtral_L32 -np 8 -smp -i pic_transient_mixtral8x7b_L32.inp -o _out_mixtral_L32.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_mixtral_L32.out && echo 'mixtral L32 OK' || echo 'mixtral L32 HAD ERRORS'
ansys252 -b -j pic_llama_L32 -np 8 -smp -i pic_transient_llamamoe_L32.inp -o _out_llama_L32.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_llama_L32.out && echo 'llama L32 OK' || echo 'llama L32 HAD ERRORS'
date
ls -la *_L4seq4096.csv *_L32.csv
echo PIC_SCALING_DONE
