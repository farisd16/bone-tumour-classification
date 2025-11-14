import torch
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm
from data.custom_dataset_class import CustomBoneDataset
import os

# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "dataset", "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "dataset", "BTXRD", "images")
results_folder = os.path.join(base_dir, "results")
os.makedirs(results_folder, exist_ok=True)


def main():

    print("Loading dataset...")
    target_classes = [
        "osteochondroma",
        "osteosarcoma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "synovial osteochondroma",
        "osteofibroma",
    ]

    dataset = CustomBoneDataset(
        image_dir=image_folder,
        json_dir=json_folder,
        image_size=128,
        grayscale=True,
        target_classes=target_classes
    )

    print(f"Number of samples found: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # === Define model and diffusion ===
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
    diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, sampling_timesteps=250, objective='pred_x0')

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=8e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion.to(device)

    print("Starting training loop...")
    for epoch in range(1):
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            loss = diffusion(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "ddpm_bone_baseline.pt")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support() 
    main()