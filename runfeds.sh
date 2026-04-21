#!/bin/bash --login

##SBATCH -p multicore  # Required (all jobs) - see table above
##SBATCH -n 32    # Required (parallel jobs)  - defaults to 1 for serial
##SBATCH -t 4-0

##SBATCH -p gpuA  # v100 GPUs
##SBATCH -G 1       # 1 GPU
##SBATCH -t 4-0       # Job will run for at most 5 minutes
##SBATCH -n 12  # (or --ntasks=) Optional number of cores. The amount of host RAM
##                  available to your job is affected by this setting.

#SBATCH -p gpuL  #
#SBATCH -G 1       # 1 GPU
#SBATCH -t 4-0       # Job will run for at most 5 minutes
#SBATCH -n 12       # (or --ntasks=) Optional number of cores. The amount of host RAM

# ------------- runtime section -------------
DATE=${DATE:-$(date +%Y%m%d)}   # fallback if --export was forgotten
LOGDIR=$HOME/scratch/FedSurrogate Backdoor Defense in Federated Learning/new_log


LOGFILE="FedS_dir0.5_MCR0.2_PDR0.3_CBA_${DATE}_${SLURM_JOB_ID}.txt"

# send both stdout and stderr to the dated file
exec >"${LOGDIR}/${LOGFILE}" 2>&1

#module purge
## Load Anaconda
#module load apps/binapps/anaconda3/2021.11

module load apps/binapps/anaconda3/2024.10  # Python 3.12.7


module load apps/binapps/pytorch/2.3.0-311-gpu-cu121
#
## Activate your Conda environment
source activate myclone_safefl

#nvdia-smi



python main.py --seed 69  --config config/mnist.yaml

