#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=30GB
#PBS -N lstm_maxEpoch_7
module purge

module load torch-deps/7
module load torch/intel/20151009

cd /home/ql516/m2k_dl/assignment3
th main.lua
