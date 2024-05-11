#!/bin/bash -l
#SBATCH --job-name=warhol
#SBATCH --time=04:00:00              
#SBATCH --output=logs/job_output_%j.log    # %j will insert the job ID
#SBATCH --gpus=1
#SBATCH --partition drl
#SBATCH -q drl-main
#SBATCH --mem-per-cpu=256G
date
echo "Running my job on $HOSTNAME"
pip3 install deeplake
python3 src/data_generation.py


