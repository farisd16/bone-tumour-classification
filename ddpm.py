import os, shutil
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import json
import shutil

def main():
    # Folders
    json_folder = "dataset/BTXRD/Annotations"
    image_folder = "dataset/BTXRD/images"
    output_folder = "filtered_data"

    os.makedirs(output_folder, exist_ok=True)

    # Classes to keep
    classes = {
        "osteochondroma",
        "osteosarcoma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "synovial osteochondroma",
        "osteofibroma",
    }

    # List JSONs
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    count = 0

    for json_name in json_files:

        # Full path to JSON
        json_path = os.path.join(json_folder, json_name)

        # Infer image filename
        image_name = json_name.replace(".json", ".jpeg")
        image_path = os.path.join(image_folder, image_name)

        # Load JSON and read label
        with open(json_path, "r") as f:
            data = json.load(f)

        label = data["shapes"][0]["label"].lower()

        # Skip if class not in the target set
        if label not in classes:
            continue

        # Skip if missing image
        if not os.path.isfile(image_path):
            print(f"⚠️ Missing image for {json_name}")
            continue

        # Copy selected image
        shutil.copy(image_path, output_folder)
        count += 1

    print(f"Finished: copied {count} images to '{output_folder}'")

    # ==== Diffusion model ====
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1  # grayscale
    )


    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # ==== Training loop ====

    trainer = Trainer(
        diffusion,
        'filtered_data',
        train_batch_size=8,
        gradient_accumulate_every=2,
        train_lr=2e-4,
        train_num_steps=70000,
        save_and_sample_every=5000,
        results_folder='./results',
        amp=True
    )


    trainer.train()


if __name__ == "__main__":
    main()