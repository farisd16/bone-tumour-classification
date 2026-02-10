#!/bin/bash
#SBATCH --job-name=dreambooth-lora-sd-1-5
#SBATCH --output=./dreambooth-lora-sd-1-5-%A.out
#SBATCH --error=./dreambooth-lora-sd-1-5-%A.err
#SBATCH --qos=master-queuesave
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Usage: sbatch finetune_dreambooth.sh

# Load python module
ml python/anaconda3

# Activate corresponding environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bone-tumour-classification

# Configuration
export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export RESOLUTION=512
export LORA_RANK=64
export DATE=$(date +%Y-%m-%d)

# All tumor classes
TUMOR_CLASSES=(
    "osteochondroma"
    "osteosarcoma"
    "multiple_osteochondromas"
    "simple_bone_cyst"
    "giant_cell_tumor"
    "synovial_osteochondroma"
    "osteofibroma"
)

BASE_DIR="/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/bone-tumour-classification/data/dataset/dreambooth"
CLASS_DIR="${BASE_DIR}/healthy"

for TUMOR_CLASS in "${TUMOR_CLASSES[@]}"; do
    echo "=============================================="
    echo "Training LoRA for: ${TUMOR_CLASS}"
    echo "=============================================="
    
    export INSTANCE_DIR="${BASE_DIR}/${TUMOR_CLASS}"
    export OUTPUT_DIR="dreambooth-lora-${TUMOR_CLASS}-rank${LORA_RANK}-${DATE}"
    
    # Unique token (sks) binds to this specific tumor type
    export INSTANCE_PROMPT="an x-ray image of sks ${TUMOR_CLASS} bone tumor"
    export CLASS_PROMPT="an x-ray image of bone"
    
    export WANDB_NAME="dreambooth-lora-${TUMOR_CLASS}-${DATE}"
    
    accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_text_encoder \
        --instance_data_dir=$INSTANCE_DIR \
        --class_data_dir=$CLASS_DIR \
        --output_dir=$OUTPUT_DIR \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --instance_prompt="$INSTANCE_PROMPT" \
        --class_prompt="$CLASS_PROMPT" \
        --resolution=$RESOLUTION \
        --rank=$LORA_RANK \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --learning_rate=1e-4 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --mixed_precision=bf16 \
        --seed=42 \
        --report_to="wandb"
    
    echo "Finished training: ${TUMOR_CLASS}"
    echo ""
done

echo "=============================================="
echo "All tumor classes trained!"
echo "=============================================="
