import os
import argparse
from datetime import datetime
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
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

MODEL_BASE_MAP = {
    "stable-diffusion": "sd-legacy/stable-diffusion-v1-5",
    "roentgen": "stanfordmimi/RoentGen-v2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmentation X-ray images using LoRA fine-tuned diffusion models"
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
    parser.add_argument(
        "--anatomical_location",
        type=str,
        default=None,
        choices=ANATOMICAL_LOCATIONS,
        help=f"Anatomical location (optional). Choices: {ANATOMICAL_LOCATIONS}",
    )
    parser.add_argument(
        "--tumor_subtype",
        type=str,
        required=True,
        choices=TUMOR_SUBTYPES,
        help=f"Tumor subtype. Choices: {TUMOR_SUBTYPES}",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=0.95,
        help="LoRA scale for generation (default: 0.95)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        choices=VIEWS,
        help=f"View type. If not specified, generates images for all views. Choices: {VIEWS}",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 25)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation (default: 7.5)",
    )
    return parser.parse_args()


def load_pipeline(model_base, lora_model_path):
    """Load the Stable Diffusion pipeline with LoRA weights."""
    pipe = DiffusionPipeline.from_pretrained(model_base, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_model_path)
    pipe.to("cuda")
    return pipe


def generate_prompt(tumor_subtype, anatomical_location=None, view=None):
    """Generate the prompt string."""
    prompt = f"X-ray image of {tumor_subtype}"
    if anatomical_location:
        prompt += f" in the {anatomical_location}"
    if view:
        prompt += f", {view} view"
    return prompt


def generate_image(
    pipe, prompt, num_inference_steps=25, guidance_scale=7.5, lora_scale=0.95
):
    """Generate a single image from a prompt."""
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]
    return image


def main():
    args = parse_args()

    model_base = MODEL_BASE_MAP[args.model_base]
    lora_model_path = args.lora_model_path

    # Output directory - same naming convention as generate_collage.py
    path_directories = lora_model_path.strip("/").split("/")

    # Create directory name with tumor/location/view combination
    safe_tumor = args.tumor_subtype.replace(" ", "_")
    safe_location = (
        args.anatomical_location.replace("-", "_") if args.anatomical_location else None
    )
    safe_view = args.view.replace(" ", "_") if args.view else None

    dir_parts = [path_directories[-2], path_directories[-1], "augmentation", safe_tumor]
    if safe_location:
        dir_parts.append(safe_location)
    if safe_view:
        dir_parts.append(safe_view)
    timestamp = datetime.now().strftime("%H-%M-%S")
    dir_parts.append(timestamp)
    output_dir = os.path.join("./generated_images", "_".join(dir_parts))
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline
    print("Loading Stable Diffusion pipeline...")
    print(lora_model_path)
    pipe = load_pipeline(model_base, lora_model_path)

    # Determine views to generate
    if args.view:
        views_to_generate = [args.view]
    elif args.view is None:
        # No view specified - generate without view in prompt
        views_to_generate = [None]

    # Calculate images per view
    images_per_view = args.num_images // len(views_to_generate)
    remainder = args.num_images % len(views_to_generate)

    print(f"\n{'=' * 60}")
    print(f"Generating {args.num_images} images")
    print(f"Tumor subtype: {args.tumor_subtype}")
    print(f"Anatomical location: {args.anatomical_location or 'not specified'}")
    print(
        f"Views: {views_to_generate if views_to_generate != [None] else 'not specified'}"
    )
    print(f"LoRA scale: {args.lora_scale}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    total_generated = 0
    for view_idx, view in enumerate(views_to_generate):
        # Distribute remainder images to first few views
        num_for_this_view = images_per_view + (1 if view_idx < remainder else 0)

        if num_for_this_view == 0:
            continue

        prompt = generate_prompt(args.tumor_subtype, args.anatomical_location, view)
        print(f"Prompt: {prompt}")

        for i in tqdm(
            range(num_for_this_view),
            desc=f"Generating{' ' + view if view else ''} images",
        ):
            try:
                image = generate_image(
                    pipe,
                    prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    lora_scale=args.lora_scale,
                )

                # Save image
                filename_parts = [safe_tumor]
                if safe_location:
                    filename_parts.append(safe_location)
                if view:
                    filename_parts.append(view.replace(" ", "_"))
                filename_parts.append(f"{total_generated:04d}")
                filename = "_".join(filename_parts) + ".png"
                image.save(os.path.join(output_dir, filename))
                total_generated += 1

            except Exception as e:
                print(f"Error generating image {i} for '{prompt}': {e}")

    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total_generated} images.")
    print(f"Images saved in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
