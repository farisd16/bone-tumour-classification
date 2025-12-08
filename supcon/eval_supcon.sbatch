#!/bin/bash

#SBATCH --job-name=eval_supcon      # Job name
#SBATCH --output=/vol/miltank/users/cahu/bone-tumour-classification/logs/eval_supcon_%j.out 
#SBATCH --error=/vol/miltank/users/cahu/bone-tumour-classification/logs/eval_supcon_%j.err
#SBATCH --partition=universe,asteroids 
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=16G                    # Request 16GB of memory
#SBATCH --time=02:00:00              # Time limit hrs:min:sec


# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate conda env
conda activate /meta/users/cahu/my_env

# Run the python script
python eval_supcon.py
