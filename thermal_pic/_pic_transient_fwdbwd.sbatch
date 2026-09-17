#!/bin/bash
#SBATCH -J pic_transient_fwdbwd
#SBATCH -A gts-syu334-ece
#SBATCH -q inferno
#SBATCH -N1 -n8
#SBATCH --mem=16G
#SBATCH -t3:00:00
#SBATCH -o /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_fwdbwd.out
#SBATCH -e /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_fwdbwd.err
cd /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic
module load ansys/2025R2
rm -f *.lock
date
ansys252 -b -j pic_mixtral_fwdbwd -np 8 -smp -i pic_transient_mixtral8x7b_fwdbwd.inp -o _out_mixtral_fwdbwd.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_mixtral_fwdbwd.out && echo 'mixtral OK (0 errors)' || echo 'mixtral HAD ERRORS'
ansys252 -b -j pic_llama_fwdbwd -np 8 -smp -i pic_transient_llamamoe_fwdbwd.inp -o _out_llama_fwdbwd.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_llama_fwdbwd.out && echo 'llama OK (0 errors)' || echo 'llama HAD ERRORS'
date
ls -la *_fwdbwd.csv
echo PIC_FWDBWD_DONE
