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
# Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <models_dir> <num_images> [output_dir]
# Example: sbatch generate_augmentation_images_dreambooth.sh stable-diffusion /path/to/dreambooth/models 100 /path/to/output
#
# The script expects DreamBooth model folders named like: dreambooth-lora-<tumor_subtype>-rank64-<date>
# It will automatically find the model for each tumor subtype and generate images.
#
# Note: This script is for DreamBooth models and always uses the 'sks' token in prompts.
#       It does NOT use detailed prompts (no anatomical location/view sampling).

MODEL_BASE=${1:?"Error: MODEL_BASE not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <models_dir> <num_images> [output_dir]"}
MODELS_DIR=${2:?"Error: MODELS_DIR not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <models_dir> <num_images> [output_dir]"}
NUM_IMAGES=${3:?"Error: NUM_IMAGES not provided. Usage: sbatch generate_augmentation_images_dreambooth.sh <model_base> <models_dir> <num_images> [output_dir]"}
OUTPUT_DIR=${4:-""}  # Optional output directory

# All tumor subtypes to generate
TUMOR_SUBTYPES=(
    "osteochondroma"
    "osteosarcoma"
    "multiple_osteochondromas"
    "simple_bone_cyst"
    "giant_cell_tumor"
    "synovial_osteochondroma"
    "osteofibroma"
)

# Build the output dir flag
OUTPUT_DIR_FLAG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_FLAG="--output_dir $OUTPUT_DIR"
fi

# Loop through all tumor subtypes
for TUMOR_SUBTYPE in "${TUMOR_SUBTYPES[@]}"; do
    echo "========================================"
    echo "Processing tumor subtype: $TUMOR_SUBTYPE"
    echo "========================================"
    
    # Find the model folder for this tumor subtype (matches dreambooth-lora-<tumor>-*)
    LORA_MODEL_PATH=$(find "$MODELS_DIR" -maxdepth 1 -type d -name "dreambooth-lora-${TUMOR_SUBTYPE}-*" | head -n 1)
    
    if [ -z "$LORA_MODEL_PATH" ]; then
        echo "WARNING: No model found for $TUMOR_SUBTYPE in $MODELS_DIR, skipping..."
        continue
    fi
    
    echo "Found model: $LORA_MODEL_PATH"
    
    srun python generate_augmentation_images.py \
        --lora_model_path "$LORA_MODEL_PATH" \
        --model_base $MODEL_BASE \
        --num_images $NUM_IMAGES \
        --tumor_subtype $TUMOR_SUBTYPE \
        --use_sks_token \
        $OUTPUT_DIR_FLAG
    
    echo "Completed: $TUMOR_SUBTYPE"
    echo ""
done

echo "========================================"
echo "All tumor subtypes processed!"
echo "========================================"
