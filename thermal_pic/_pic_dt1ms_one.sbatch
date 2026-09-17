#!/bin/bash
#SBATCH -A gts-syu334-ece
#SBATCH -q inferno
#SBATCH -N1 -n8
#SBATCH --mem=16G
#SBATCH -t3:00:00
# usage: sbatch -J <job> -o logs_dt1ms/<job>.out -e logs_dt1ms/<job>.err _pic_dt1ms_one.sbatch <stem>
# <stem> = pic_transient_<model>_<variant>_dt1ms (without .inp)
set -u
STEM=$1
cd /storage/home/hcoda1/8/syoon351/scratch/repos/jetc-rev/mixnet-sim/thermal_pic
module load ansys/2025R2
JOB=${STEM#pic_transient_}
rm -f ${JOB}.lock
date
ansys252 -b -j "$JOB" -np 8 -smp -i "$STEM.inp" -o "logs_dt1ms/_out_${JOB}.out"
grep -q "NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0" "logs_dt1ms/_out_${JOB}.out" && echo "$JOB OK" || echo "$JOB HAD ERRORS"
date
