# generate_fid_samples.py

import os
import random
from pathlib import Path

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import trange
import torch

# -------------------------
# Configuration
# -------------------------
# Number of images to generate
N_SAMPLES = 1000


# Where to save the generated images
OUTPUT_DIR = (
    "/vol/miltank/users/carre/"
    "bone-tumour-classification/latent_diffusion_finetuned/fid_evaluation_samples/fid_1000_samples_0_7"
)

# Diffusion + LoRA config (adapt paths if needed)
MODEL_BASE = "sd-legacy/stable-diffusion-v1-5"
LORA_MODEL_PATH = (
    "/vol/miltank/users/carre/bone-tumour-classification/"
    "latent_diffusion_finetuned/lora_weights/"
    "sd-1-5-lora-rank-32-batch-4-resolution-512/checkpoint-5000"
)


NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
LORA_SCALE = 0.7  # [0.7 to 1]

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
# Pipeline helpers
# -------------------------


def load_pipeline():
    """Load the Stable Diffusion pipeline with LoRA weights."""
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_BASE, use_safetensors=True, safety_checker=None
    )
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
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading Stable Diffusion pipeline with LoRA...")
    pipe = load_pipeline()

    print(f"Generating {N_SAMPLES} images into: {OUTPUT_DIR}")
    for i in trange(N_SAMPLES, desc="Generating FID samples"):
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

            image.save(os.path.join(OUTPUT_DIR, filename))
        except Exception as e:
            print(f"Error at index {i} for prompt '{prompt}': {e}")

    print("Done. All FID samples generated.")


if __name__ == "__main__":
    main()
