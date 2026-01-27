#!/bin/bash
#SBATCH --job-name=generate-augmentation-images
#SBATCH --output=./generate-augmentation-images-%A.out
#SBATCH --error=./generate-augmentation-images-%A.err
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
# Usage: sbatch generate_augmentation_images.sh <model_base> <lora_model_path> <num_images> <tumor_subtype>
# Example: sbatch generate_augmentation_images.sh stable-diffusion /path/to/lora/weights 100 osteofibroma
MODEL_BASE=${1:?"Error: MODEL_BASE not provided. Usage: sbatch generate_augmentation_images.sh <model_base> <lora_model_path> <num_images> <tumor_subtype>"}
LORA_MODEL_PATH=${2:?"Error: LORA_MODEL_PATH not provided. Usage: sbatch generate_augmentation_images.sh <model_base> <lora_model_path> <num_images> <tumor_subtype>"}
NUM_IMAGES=${3:?"Error: NUM_IMAGES not provided. Usage: sbatch generate_augmentation_images.sh <model_base> <lora_model_path> <num_images> <tumor_subtype>"}
TUMOR_SUBTYPE=${4:?"Error: tumor_subtype not provided. Usage: sbatch generate_augmentation_images.sh <model_base> <lora_model_path> <num_images> <tumor_subtype>"}

srun python generate_augmentation_images.py \
    --lora_model_path $LORA_MODEL_PATH \
    --model_base $MODEL_BASE \
    --num_images $NUM_IMAGES \
    --tumor_subtype $TUMOR_SUBTYPE

    