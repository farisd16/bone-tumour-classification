# generate_fid_samples.py

import os
import random
from pathlib import Path
import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import trange
import torch

# -------------------------
# Configuration
# -------------------------
# Number of images to generate
DEFAULT_N_SAMPLES = 50000

DEFAULT_OUTPUT_DIR = (
    "/vol/miltank/users/carre/"
    "bone-tumour-classification/latent_diffusion_finetuned/fid_evaluation_samples/"
    "fid_50000_samples_lora_1"
)

DEFAULT_MODEL_BASE = "sd-legacy/stable-diffusion-v1-5"

DEFAULT_LORA_MODEL_PATH = (
    "/vol/miltank/users/carre/bone-tumour-classification/"
    "latent_diffusion_finetuned/lora_weights/"
    "sd-1-5-lora-rank-32-batch-4-resolution-512/checkpoint-5000"
)




NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
LORA_SCALE = 1

# Prompts
TUMOR_SUBTYPES = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

ANATOMICAL_LOCATIONS = [
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
]

VIEWS = ["frontal", "lateral", "oblique"]

# -------------------------
# Argument parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FID samples with SD + LoRA."
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default=DEFAULT_MODEL_BASE,
        help="Base Stable Diffusion model (e.g. sd-legacy/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=DEFAULT_LORA_MODEL_PATH,
        help="Path to LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of images to generate.",
    )
    return parser.parse_args()



# -------------------------
# Pipeline helpers
# -------------------------

def load_pipeline(MODEL_BASE: str, LORA_MODEL_PATH: str):
    """Load the Stable Diffusion pipeline with LoRA weights."""
    pipe = DiffusionPipeline.from_pretrained(MODEL_BASE, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LORA_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    pipe.to(device)

    return pipe


def generate_prompt():
    """Randomly sample tumor subtype, location, view and build a prompt."""
    tumor = random.choice(TUMOR_SUBTYPES)
    location = random.choice(ANATOMICAL_LOCATIONS)
    view = random.choice(VIEWS)
    prompt = f"X-ray image of {tumor} in the {location}, {view} view"
    return prompt, tumor, location, view


def main():
    args = parse_args()

    output_dir = args.output_dir
    n_samples = args.n_samples

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading Stable Diffusion pipeline with LoRA...")
    pipe = load_pipeline(args.model_base, args.lora_model_path)

    print(f"Generating {n_samples} images into: {output_dir}")
    for i in trange(n_samples, desc="Generating FID samples"):
        prompt, tumor, location, view = generate_prompt()
        try:
            image = pipe(
                prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                cross_attention_kwargs={"scale": LORA_SCALE},
            ).images[0]

            # Unique filename with index + metadata
            safe_tumor = tumor.replace(" ", "_").replace("-", "_")
            safe_loc = location.replace(" ", "_").replace("-", "_")
            safe_view = view.replace(" ", "_").replace("-", "_")
            filename = f"{i:06d}_{safe_tumor}_{safe_loc}_{safe_view}.png"

            image.save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Error at index {i} for prompt '{prompt}': {e}")

    print("Done. All FID samples generated.")


if __name__ == "__main__":
    main()