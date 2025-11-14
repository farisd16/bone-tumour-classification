import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Recreate model and diffusion (must match your training config) ---
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000, sampling_timesteps=250, objective='pred_x0').to(device)

# --- Load trained weights ---
model.load_state_dict(torch.load("ddpm_bone_baseline.pt", map_location=device))

# --- Set model to eval mode ---
model.eval()

# --- Generate synthetic samples ---
with torch.no_grad():
    sampled_images = diffusion.sample(batch_size=4)  # 4 grayscale images [4,1,64,64]

# --- Save to disk for inspection ---
save_image(sampled_images, "synthetic_bone_samples.png", nrow=2, normalize=True)

print("Synthetic samples saved as synthetic_bone_samples.png")
