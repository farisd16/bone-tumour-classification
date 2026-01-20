#!/bin/bash
#SBATCH --job-name=finetune-sd-1-5
#SBATCH --output=./finetune-sd-1-5-%A.out
#SBATCH --error=./finetune-sd-1-5-%A.err
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
# Usage: sbatch generate_collage.sh <model_base> <lora_model_path>
# Example: sbatch generate_collage.sh stable-diffusion /path/to/lora/weights
MODEL_BASE=${1:?"Error: MODEL_BASE not provided. Usage: sbatch generate_collage.sh <model_base> <lora_model_path>"}
LORA_MODEL_PATH=${2:?"Error: LORA_MODEL_PATH not provided. Usage: sbatch generate_collage.sh <model_base> <lora_model_path>"}

srun python generate_collage.py \
    --model_base=$MODEL_BASE \
    --lora_model_path=$LORA_MODEL_PATH