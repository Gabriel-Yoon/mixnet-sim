#!/bin/bash
#SBATCH -J pic_transient_qwenmoe
#SBATCH -A gts-syu334-ece
#SBATCH -q inferno
#SBATCH -N1 -n8
#SBATCH --mem=16G
#SBATCH -t3:00:00
#SBATCH -o /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_qwenmoe.out
#SBATCH -e /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic/_sbatch_pic_qwenmoe.err
cd /storage/home/hcoda1/8/syoon351/scratch/repos/mixnet-sim/thermal_pic
module load ansys/2025R2
rm -f *.lock
date
ansys252 -b -j pic_qwenmoe_L4seq4096 -np 8 -smp -i pic_transient_qwenmoe_L4seq4096.inp -o _out_qwenmoe_L4seq4096.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_qwenmoe_L4seq4096.out && echo 'qwenmoe L4seq4096 OK' || echo 'qwenmoe L4seq4096 HAD ERRORS'
ansys252 -b -j pic_qwenmoe_L24 -np 8 -smp -i pic_transient_qwenmoe_L24.inp -o _out_qwenmoe_L24.out
grep -q 'NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0' _out_qwenmoe_L24.out && echo 'qwenmoe L24 OK' || echo 'qwenmoe L24 HAD ERRORS'
date
ls -la qwenmoe_pic_transient_L4seq4096.csv qwenmoe_pic_transient_L24.csv
echo PIC_QWENMOE_DONE
