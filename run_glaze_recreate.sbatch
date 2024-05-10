#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --time=00:04:00              
#SBATCH --output=job_output_%j.log    # %j will insert the job ID
#SBATCH --gpus=1
#SBATCH --partition drl
#SBATCH -q drl-main
#SBATCH --mem-per-cpu=256G
date
echo "Running my job on $HOSTNAME"
python3 src/data_generation.py


