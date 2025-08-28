#!/bin/bash
#SBATCH --time=8-00:00:00                 # Time limit for the job (REQUIRED).
#SBATCH --job-name=EELG1002             # Job name
#SBATCH --ntasks=128                      # Number of cores for the job. Same as SBATCH -n 1
#SBATCH --partition=normal                # Partition/queue to run the job in. (REQUIRED)
#SBATCH -e slurm-EELG1002.err      # Error file for this job.
#SBATCH -o slurm-EELG1002.out       # Output file for this job.
#SBATCH --account=coa_rlsa239_uksr        # Project allocation account name (REQUIRED)

# Load in the Python Environment for Bagpipes
source $PROJECT/ali_ahmad/envs/bagpipes/bin/activate

mpirun -np 128 python bagpipes_EELG_OIII_GMOS_run.py
