#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./train-%A.out
#SBATCH --error=./train-%A.err
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
conda activate bone-tumour-classification
# Run the program
# Usage: sbatch train.sh <synthetic_split> <run_name_prefix>
# Example: sbatch train.sh stable-diffusion /path/to/lora/weights 100 osteofibroma
SYNTHETIC_SPLIT=${1:?"Error: SYNTHETIC_SPLIT not provided. Usage: sbatch train.sh <synthetic_split> <run_name_prefix>"}
RUN_NAME_PREFIX=${2:?"Error: RUN_NAME_PREFIX not provided. Usage: sbatch train.sh <synthetic_split> <run_name_prefix>"}

srun python train.py \
    --trainwsyn $SYNTHETIC_SPLIT \
    --run-name-prefix $RUN_NAME_PREFIX

    