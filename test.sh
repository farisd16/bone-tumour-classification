#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=.logs/test-%A.out
#SBATCH --error=.logs/test-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that.
# Not necessary if you always run this script from a clean terminal
eval "$(conda shell.bash hook)"
conda deactivate
# If the following does not work, try 'source activate <env-name>'
conda activate ADLM
# Run the program
# Usage: sbatch test.sh <run_name>
# Example: sbatch test.sh resnet_diffusion-synthetic_split_step1_wce_aug_2026-01-22_15-59-59
RUN_NAME=${1:?"Error: RUN_NAME not provided. Usage: sbatch test.sh <run_name>"}

srun python test.py \
    --run-name $RUN_NAME

    