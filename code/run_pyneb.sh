#!/bin/bash
#SBATCH --job-name=pyneb_job
#SBATCH --account=csa
#SBATCH --output=slurm_pyneb.out
#SBATCH --error=slurm_pyneb.err
#SBATCH -p scavenger
#SBATCH --nodelist=skl-a-00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       # adjust this as needed
#SBATCH --mem=16G               # adjust as needed
#SBATCH --time=01-00:00:00         # adjust as needed

source $HOME/env/pyneb/bin/activate

srun python pyneb_measurements.py