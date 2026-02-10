# This script is in diffusers/examples/text_to_image on the server

#!/bin/bash
#SBATCH --job-name=finetune-sd-1-5
#SBATCH --output=./finetune-sd-1-5-%A.out
#SBATCH --error=./finetune-sd-1-5-%A.err
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=48:00:00
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
export RESOLUTION=512
export LORA_RANK=32
export BATCH_SIZE=4
export DATE=$(date +%Y-%m-%d)
export OUTPUT_DIR="sd-1-5-lora-rank-$LORA_RANK-batch-$BATCH_SIZE-resolution-$RESOLUTION-$DATE"
export MODEL_NAME="sd-legacy/stable-diffusion-v1-5"
export TRAIN_DIR="/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/bone-tumour-classification/data/dataset/hf_dataset"
export WANDB_NAME="sd-1-5-rank$LORA_RANK-batch$BATCH_SIZE-resolution$RESOLUTION-$DATE"
srun accelerate launch train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$TRAIN_DIR \
    --caption_column="text" \
    --resolution=$RESOLUTION \
    --rank=$LORA_RANK \
    --mixed_precision=bf16 \
    --train_batch_size=$BATCH_SIZE \
    --max_train_steps=20000 \
    --checkpointing_steps=5000 \
    --learning_rate=1e-04 \
    --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt="X-ray image of osteochondroma in the tibia, lateral view" \
    --report_to="wandb"
