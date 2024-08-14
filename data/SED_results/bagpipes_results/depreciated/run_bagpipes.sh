#!/bin/bash -l
# NOTE the -l flag!
# This is an example job file for a multi-core MPI job.
# If you need any help, please [submit a ticket](https://help.rit.edu/sp?id=rc_request) or contact us on Slack.
# Name of the job 
#SBATCH -J bagpipes
# Standard out and Standard Error output files
#SBATCH -o slurm_bagpipes.txt
#SBATCH -e error_bagpipes.txt
#Put the job in the appropriate partition matching the account and request FOUR cores
#SBATCH -p scavenger -n 20
#Job memory requirements in MB=m (default), GB=g, or TB=t
#SBATCH --mem=16g
#SBATCH --time=01-00:00:00

source ~/env/bagpipes/bin/activate
srun -n 20 python bagpipes_EELG_OIII_GMOS_run.py
