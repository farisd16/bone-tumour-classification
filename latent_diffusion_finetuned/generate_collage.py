import os
import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
from itertools import product
from tqdm import tqdm

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

LORA_SCALES = [0.7, 0.8, 0.9, 1]

MODEL_BASE_MAP = {
    "stable-diffusion": "sd-legacy/stable-diffusion-v1-5",
    "roentgen": "stanfordmimi/RoentGen-v2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate collage of X-ray images using LoRA fine-tuned diffusion models"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="Path to the LoRA weights directory",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        required=True,
        choices=["stable-diffusion", "roentgen"],
        help="Base model to use: 'stable-diffusion' or 'roentgen'",
    )
    return parser.parse_args()


def load_pipeline(model_base, lora_model_path):
    """Load the Stable Diffusion pipeline with LoRA weights."""
    pipe = DiffusionPipeline.from_pretrained(model_base, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_model_path)
    pipe.to("cuda")

    return pipe


def generate_prompt(tumor_subtype, anatomical_location, view):
    """Generate the prompt string."""
    return f"X-ray image of {tumor_subtype} in the {anatomical_location}, {view} view"


def generate_image(
    pipe, prompt, num_inference_steps=25, guidance_scale=7.5, lora_scale=0.7
):
    """Generate a single image from a prompt."""
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]
    return image


def create_collage(
    images, labels, images_per_row, image_size=(512, 512), label_height=50
):
    """Create a collage from a list of images with labels.

    Labels should be tuples of (tumor_type, anatomical_location, view).
    """
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    cell_width = image_size[0]
    cell_height = image_size[1] + label_height

    collage_width = images_per_row * cell_width
    collage_height = num_rows * cell_height

    collage = Image.new("RGB", (collage_width, collage_height), color="white")
    draw = ImageDraw.Draw(collage)

    # Calculate font size for 3 lines within label_height (with some padding)
    font_size = max(10, (label_height - 8) // 3)

    # Try to load a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except:
        font = ImageFont.load_default(size=font_size)

    line_height = font_size + 2

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // images_per_row
        col = idx % images_per_row

        x = col * cell_width
        y = row * cell_height

        # Resize image
        img_resized = img.resize(image_size, Image.Resampling.LANCZOS)

        # Paste image
        collage.paste(img_resized, (x, y))

        # Add multi-line label below image
        label_y = y + image_size[1] + 2

        # label is a tuple: (tumor_type, anatomical_location, view)
        tumor_type, location, view = label

        # Truncate each line if needed
        char_width = font_size * 0.6
        max_chars = int((cell_width - 10) / char_width)

        lines = [
            tumor_type[:max_chars] if len(tumor_type) > max_chars else tumor_type,
            location[:max_chars] if len(location) > max_chars else location,
            view[:max_chars] if len(view) > max_chars else view,
        ]

        for i, line in enumerate(lines):
            draw.text((x + 5, label_y + i * line_height), line, fill="black", font=font)

    return collage


def main():
    args = parse_args()

    model_base = MODEL_BASE_MAP[args.model_base]
    lora_model_path = args.lora_model_path

    # Output directory
    path_directories = lora_model_path.strip("/").split("/")
    base_output_dir = f"./latent_diffusion_finetuned/generated_images/{path_directories[-2]}_{path_directories[-1]}_collages"
    os.makedirs(base_output_dir, exist_ok=True)

    # Load pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = load_pipeline(model_base, lora_model_path)

    # Generate all combinations
    all_combinations = list(product(TUMOR_SUBTYPES, ANATOMICAL_LOCATIONS, VIEWS))
    total_images = len(all_combinations)
    print(f"Total combinations per LoRA scale: {total_images}")
    print(f"Total images to generate: {total_images * len(LORA_SCALES)}")

    # Generate for each LoRA scale
    for lora_scale in LORA_SCALES:
        print(f"\n{'=' * 60}")
        print(f"Generating images with LoRA scale: {lora_scale}")
        print(f"{'=' * 60}")

        # Create output directory for this scale
        output_dir = os.path.join(base_output_dir, f"lora_scale_{lora_scale}")
        os.makedirs(output_dir, exist_ok=True)

        images = []
        labels = []

        # Generate images for each combination
        for tumor, location, view in tqdm(all_combinations, desc=f"LoRA {lora_scale}"):
            prompt = generate_prompt(tumor, location, view)

            try:
                image = generate_image(pipe, prompt, lora_scale=lora_scale)
                images.append(image)
                # Create label as tuple (tumor_type, location, view)
                labels.append((tumor, location, view))

                # Save individual image
                safe_filename = f"{tumor}_{location}_{view}".replace(" ", "_").replace(
                    "-", "_"
                )
                image.save(os.path.join(output_dir, f"{safe_filename}.png"))

            except Exception as e:
                print(f"Error generating image for '{prompt}': {e}")
                # Create a placeholder image
                placeholder = Image.new("RGB", (512, 512), color="gray")
                images.append(placeholder)
                labels.append((f"ERROR: {tumor}", location, view))

        # Create collage
        print(f"Creating collage for LoRA scale {lora_scale}...")
        collage = create_collage(
            images, labels, images_per_row=15, image_size=(128, 128), label_height=45
        )

        # Save collage
        collage_path = os.path.join(output_dir, f"collage_lora_{lora_scale}.png")
        collage.save(collage_path)
        print(f"Collage saved to: {collage_path}")

        # Also save to base directory for easy comparison
        collage.save(os.path.join(base_output_dir, f"collage_lora_{lora_scale}.png"))

    print("\n" + "=" * 60)
    print("Done! All collages generated.")
    print(f"Collages saved in: {base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
