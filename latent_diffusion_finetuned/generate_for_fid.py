import os
from itertools import islice, product

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm

# ---------------------------
# 1) PROMPT COMPONENTS
#    (copied from generate_collage.py)
# ---------------------------

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

# ---------------------------
# 2) MODEL CONFIG
# ---------------------------
# You can switch between SD 1.5 and RoentGen.
# Here we use your RoentGen setup like in generate_collage.py.

# model_base = "stanfordmimi/RoentGen-v2"
model_base = "sd-legacy/stable-diffusion-v1-5"

# Path to your LoRA checkpoint (adjust if your folder name differs)
# lora_model_path = (
#     "./latent_diffusion_finetuned/lora_weights/"
#     "roentgen-btxrd-model-lora-rank-32-batch-4-resolution-512"
# )

lora_model_path = "./latent_diffusion_finetuned/lora_weights/sd-1-5-btxrd-model-lora-rank-32-batch-4/checkpoint-5000"


# Where to save the generated images (absolute path recommended)
OUT_DIR = (
    "/vol/miltank/projects/practical_wise2526/"
    "bone-tumor-classification-gen-models/btxrd_fid/fake_ldm_train"
)


# ---------------------------
# 3) PIPELINE LOADING
# ---------------------------

def load_pipeline():
    """Load the diffusion pipeline with LoRA weights on GPU."""
    print("Loading pipeline...")
    pipe = DiffusionPipeline.from_pretrained(model_base, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_model_path)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline loaded.")
    return pipe


# ---------------------------
# 4) PROMPT BUILDING
# ---------------------------

def generate_prompt(tumor_subtype, anatomical_location, view):
    """Generate the BTXRD-style text prompt."""
    return f"X-ray image of {tumor_subtype} in the {anatomical_location}, {view} view"


# ---------------------------
# 5) MAIN GENERATION FUNCTION
# ---------------------------

def main(num_images=10000):
    """
    Generate num_images synthetic BTXRD-style images and save to OUT_DIR.

    You can start with num_images=2000 to test, then increase to 10000 or 50000.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Images will be saved to: {OUT_DIR}")

    pipe = load_pipeline()

    # All combinations of tumor, location, view
    all_combos = list(product(TUMOR_SUBTYPES, ANATOMICAL_LOCATIONS, VIEWS))

    # If there are more combos than num_images, only take the first num_images
    combos = list(islice(all_combos, num_images))

    print(f"Total combinations available: {len(all_combos)}")
    print(f"Number of images to generate: {len(combos)}")

    for i, (tumor, location, view) in enumerate(tqdm(combos, desc="Generating")):
        prompt = generate_prompt(tumor, location, view)

        try:
            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                cross_attention_kwargs={"scale": 1.0},
            ).images[0]

            # Ensure RGB and 512x512
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((512, 512), Image.BICUBIC)

            filename = f"fake_{i:06d}.png"
            image.save(os.path.join(OUT_DIR, filename))

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            # Optional: create a gray placeholder so indexing stays consistent
            placeholder = Image.new("RGB", (512, 512), color="gray")
            filename = f"fake_{i:06d}.png"
            placeholder.save(os.path.join(OUT_DIR, filename))

    print("Done! All fake images generated.")


if __name__ == "__main__":
    # You can change this number if needed
    main(num_images=10000)
