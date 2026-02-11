#!/bin/bash
#SBATCH --job-name=generate-dreambooth-images
#SBATCH --output=./generate-dreambooth-images-%A.out
#SBATCH --error=./generate-dreambooth-images-%A.err
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load python module
ml python/anaconda3

# Activate corresponding environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bone-tumour-classification

# Run the program
# Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <lora_model_path> <num_images> <tumor_subtype> [output_dir]
# Example: sbatch generate_augmentation_images_dreambooth.sh stable-diffusion /path/to/dreambooth/weights 100 osteofibroma /path/to/output
# Example with all subtypes: sbatch generate_augmentation_images_dreambooth.sh stable-diffusion /path/to/dreambooth/weights 100 all /path/to/output
#
# Note: This script is for DreamBooth models and always uses the 'sks' token in prompts.
#       It does NOT use detailed prompts (no anatomical location/view sampling).

MODEL_BASE=${1:?"Error: MODEL_BASE not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <lora_model_path> <num_images> <tumor_subtype> [output_dir]"}
LORA_MODEL_PATH=${2:?"Error: LORA_MODEL_PATH not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <lora_model_path> <num_images> <tumor_subtype> [output_dir]"}
NUM_IMAGES=${3:?"Error: NUM_IMAGES not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <lora_model_path> <num_images> <tumor_subtype> [output_dir]"}
TUMOR_SUBTYPE=${4:?"Error: tumor_subtype not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <lora_model_path> <num_images> <tumor_subtype> [output_dir]"}
OUTPUT_DIR=${5:-""}  # Optional output directory

# Build the output dir flag
OUTPUT_DIR_FLAG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_FLAG="--output_dir $OUTPUT_DIR"
fi

srun python generate_augmentation_images.py \
    --lora_model_path $LORA_MODEL_PATH \
    --model_base $MODEL_BASE \
    --num_images $NUM_IMAGES \
    --tumor_subtype $TUMOR_SUBTYPE \
    --use_sks_token \
    $OUTPUT_DIR_FLAG
