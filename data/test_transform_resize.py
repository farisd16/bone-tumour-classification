import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(project_root, "data", "patched_BTXRD")
output_folder = os.path.join(project_root, "data", "resized_BTXRD")
target_size = (224, 224)

os.makedirs(output_folder, exist_ok=True)

resize_transform = transforms.Compose(
    [
        transforms.Resize(target_size),
        # Optional, should retain aspect ratio: transforms.CenterCrop(224)
    ]
)

for filename in tqdm(os.listdir(input_folder), desc="Resizing images"):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    try:
        img = Image.open(input_path)
        resized_img = resize_transform(img)
        resized_img.save(output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
