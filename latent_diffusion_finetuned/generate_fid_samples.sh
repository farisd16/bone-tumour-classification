#!/bin/bash
#SBATCH --job-name=fid-samples
#SBATCH --output=fid-samples-%j.out
#SBATCH --error=fid-samples-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# export HF_TOKEN=your_token_here
# export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
# unset HF_HOME
# export HF_HOME=/vol/miltank/users/carre/.hf
# export HF_HUB_CACHE=$HF_HOME/hub
# export TRANSFORMERS_CACHE=$HF_HOME/transformers

PROJECT_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned
ENV_NAME=bone-tumour-classification


# -------------------------
# Configuration
# -------------------------
MODEL_BASE="sd-legacy/stable-diffusion-v1-5" #"stanfordmimi/RoentGen-v2"
LORA_MODEL_PATH="$PROJECT_DIR/lora_weights/sd-1-5-lora-rank-64-batch-4-resolution-512-2026-02-09/checkpoint-20000"
OUTPUT_DIR="$PROJECT_DIR/fid_evaluation_samples/rep_stable_diffusion_rank_64_batch_4_v1_5_lora_1_ckp_20000"
N_SAMPLES=1000


cd "$PROJECT_DIR"

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Fix SSL certs for HF Hub (SLURM/conda issue)
conda install -c conda-forge openssl ca-certificates certifi -y
export SSL_CERT_FILE=$(conda run -n $ENV_NAME python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"

pip install -U transformers diffusers


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi


python - << 'EOF'
import torch
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
EOF


# Run with parsed arguments
python generate_fid_samples.py \
    --model_base "$MODEL_BASE" \
    --lora_model_path "$LORA_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES"

