#!/bin/bash
#SBATCH --job-name=parse-diff-images
#SBATCH --output=./parse-diff-images-%A.out
#SBATCH --error=./parse-diff-images-%A.err
##SBATCH --partition=asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Load Python module and activate conda env
ENV_NAME=bone-tumour-classification

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Repo dir
REPODIR="/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned"
cd "$REPODIR"

INPUT_DIR="${INPUT_DIR:-generated_images_flat}"  # Export INPUT_DIR=/path/to/flat/images
OUTPUT_DIR="${OUTPUT_DIR:-generated_images_organized}"  # Export OUTPUT_DIR=/path/to/output

export INPUT_DIR=./fid_evaluation_samples/rep_stable_diffusion_rank_64_batch_4_v1_5_lora_1_ckp_20000
export OUTPUT_DIR=./organized_evaluation_samples/rep_stable_diffusion_rank_64_batch_4_v1_5_lora_1_ckp_20000

# Tumor subtypes (spaces -> underscores)
SUBTYPES=(
  "osteochondroma"
  "osteosarcoma"
  "multiple_osteochondromas"
  "simple_bone_cyst"
  "giant_cell_tumor"
  "synovial_osteochondroma"
  "osteofibroma"
)

# Gross location mapping (substring match on location_detail)
declare -A LOCATION_MAP=(
  ["hand"]="upper_limb"
  ["ulna"]="upper_limb"
  ["radius"]="upper_limb"
  ["humerus"]="upper_limb"
  ["wrist_joint"]="upper_limb"
  ["elbow_joint"]="upper_limb"
  ["shoulder_joint"]="upper_limb"
  ["foot"]="lower_limb"
  ["tibia"]="lower_limb"
  ["fibula"]="lower_limb"
  ["femur"]="lower_limb"
  ["ankle_joint"]="lower_limb"
  ["knee_joint"]="lower_limb"
  ["hip_bone"]="pelvis"
  ["hip_joint"]="pelvis"
)

mkdir -p "$OUTPUT_DIR"

# Process all PNG images
for img in "$INPUT_DIR"/*.png; do
  [[ ! -f "$img" ]] && continue
  base=$(basename "$img" .png)

  view="none"
  base_no_view="$base"

  if [[ "$base" =~ _(frontal|lateral|oblique)$ ]]; then
    view="${BASH_REMATCH[1]}"
    base_no_view="${base%_$view}"
  fi

  # Remove leading digits_
  rest="${base_no_view#[0-9]*_}"

  raw_subtype=""
  location_detail=""

  # Match subtype by known list (longest match wins)
  for st in "${SUBTYPES[@]}"; do
    if [[ "$rest" == "$st"_* ]]; then
      raw_subtype="$st"
      location_detail="${rest#${st}_}"
      break
    fi
  done

  norm_subtype="$(echo "$raw_subtype" | tr '[:upper:]' '[:lower:]')"
  #location_detail="${location_detail//_/-}"

  gross_location="unknown"
  for key in "${!LOCATION_MAP[@]}"; do
    if [[ "$location_detail" == *"$key"* ]]; then
      gross_location="${LOCATION_MAP[$key]}"
      break
    fi
  done

  target_subtype=""
  for st in "${SUBTYPES[@]}"; do
    if [[ "$norm_subtype" == "$st" ]]; then
      target_subtype="$st"
      break
    fi
  done

  if [[ -n "$target_subtype" && "$gross_location" != "unknown" ]]; then
    target_dir="$OUTPUT_DIR/${target_subtype}_${gross_location}"
    mkdir -p "$target_dir"
    mv "$img" "$target_dir/"
    echo "Moved $img -> $target_dir"
  else
    echo "Skipped $img: subtype='$norm_subtype' location='$location_detail' -> gross='$gross_location'"
  fi
done

echo "Organization complete. Check $OUTPUT_DIR for 21 folders (7 subtypes x 3 locations)."
ls -l "$OUTPUT_DIR" | head -21
