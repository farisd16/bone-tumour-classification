# 🦴 **ADLM: Addressing Class Imbalance in Bone Tumor X-Ray Classification with Generative Models**

This project explores how **generative models** can help address **class imbalance** in **bone tumor X-ray classification**.  
We use the **BTXRD dataset**, which contains X-ray images of different primary bone tumor entities.

---

## 🎯 Project Goals

- Implement a **ResNet-based CNN** as a **baseline model** for tumor classification.
- Apply and analyze **class imbalance handling techniques** such as **Weighted Loss**, **Focal Loss**, and **Data Augmentation**.
- Use **generative models** (e.g., **GANs** or **Diffusion Models**) to create synthetic X-ray images and evaluate their impact on classification performance.

---

## 🦴 Target Tumor Types

We focus on classifying **seven tumor types**:

- Osteochondroma
- Osteosarcoma
- Multiple Osteochondromas
- Simple Bone Cyst
- Giant Cell Tumor
- Synovial Osteochondroma
- Osteofibroma

---

## 🏫 About the Project

This project is part of the **Applied Deep Learning in Medicine (ADLM)** course at the  
**Technical University of Munich (TUM)**, organized by the
**Chair of Artificial Intelligence in Healthcare and Medicine**.

## ⬇️ Dependencies

First, create and activate a virtual environment:

**Option 1: Using Conda**

```bash
conda create -n bone-tumour-classification python=3.12
conda activate bone-tumour-classification
```

**Option 2: Using venv**
x

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

Then install dependencies:

```
pip install -r requirements.txt
```

---

## ⬇️ Dataset Setup

You can download the BTXRD dataset [here](https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398?file=50653575)

Place the BTXRD dataset under the following paths relative to the project root:

```
data/
  dataset/
    BTXRD/
      images/         # Original X-ray images (e.g., IMG000123.jpeg)
      Annotations/    # JSON annotation files (same basenames as images)
```

(Optional) folders created by scripts in this repo:

```
data/
  dataset/
    final_patched_BTXRD/    # Extracted patches from annotations (created, this is the final dataset used for training and testing)
    squared_padded/         # Padded originals for 106 special cases (created) (optional)
```

### Data Validation & Debugging Utilities

The following files were used to identify and resolve issues that occurred during the preprocessing of the BTXRD dataset. These validation and debugging scripts ensured data integrity and enabled the creation of the final `final_patched_BTXRD` dataset.

- `bounding_box_checker.py`  
  Adds the images, whose bounding box exceeds the original image size, to the csv file. Tells also whether the bounding box of that image
  exceeds the image size
- `bounding_box_visualization.py`  
  Helps to visualize bounding boxes
- `tumour_bounding_box.py`  
  Function that computes a square bounding box (with optional margin) around all given tumour points.
- `pad_unsquared.py`  
  Creates the squared padded folder in the dataset folder, containing padded images with their pad info in the csv file

## 📊 Classification

### ▶️ How To Run (Preparation → Training → Testing)

1. Extract patches from annotations

```bash
python data/btxrd_bounding_box_dataset_extractor.py
```

This creates `data/dataset/final_patched_BTXRD/` from `BTXRD/images` + `BTXRD/Annotations`.

2. Train classification model

```bash
python train.py [arguments]
```

Arguments:

- `--learning-rate` (`--learning_rate`, float, default `9.502991994821847e-05`)
- `--weight-decay` (`--weight_decay`, float, default `1e-05`)
- `--batch-size` (`--batch_size`, int, default `32`)
- `--epochs` (int, default `30`)
- `--dropout` (float, default `0.3498137514984224`)
- `--scheduler-factor` (`--scheduler_factor`, float, default `0.5`)
- `--scheduler-patience` (`--scheduler_patience`, int, default `4`)
- `--test-size` (`--test_size`, float, default `0.2`)
- `--random-state` (`--random_state`, int, default `42`)
- `--loss-fn` (`--loss_fn`, choices: `ce|wce|focal|wfocal`, default `wce`)
- `--focal-gamma` (`--focal_gamma`, float, default `2.924897740591147`)
- `--apply-minority-aug` (`--apply_minority_aug`, bool, default `False`)
- `--early-stop` (`--early_stop`, bool, default `False`)
- `--early-stop-patience` (`--early_stop_patience`, int, default `5`)
- `--early-stop-min-delta` (`--early_stop_min_delta`, float, default `0.0`)
- `--run-name-prefix` (`--run_name_prefix`, str, default `resnet_gan_15800`)
- `--num-classes` (`--num_classes`, int, default `7`)
- `--architecture` (choices: `resnet34|resnet50|densenet121`, default `resnet34`)
- `--trainwsyn` (str path to split JSON, default `None`)
- `--image-dir` (`--image_dir`, str, default `data/dataset/final_patched_BTXRD`)
- `--json-dir` (`--json_dir`, str, default `data/dataset/BTXRD/Annotations`)

Example:

```bash
python train.py --architecture resnet50 --loss-fn wce --early-stop true --early-stop-patience 10 --early-stop-min-delta 0.001 --batch-size 32 --epochs 50
```

Notes:

- Labels are read directly from annotation JSON files.
- Checkpoints are saved under `checkpoints/<run_name>/`.

3. Test trained model

```bash
python test.py --run-name <RUN_NAME> [--architecture resnet34|resnet50|densenet121]
```

Arguments:

- `--run-name` (required): Run/checkpoint folder name under `checkpoints/` and matching W&B display name.
- `--architecture` (optional): Backbone used during training. Default is `resnet34`.

Example:

```bash
python test.py --run-name resnet_gan_15800_wce_noaug_2026-02-12_14-30-00 --architecture resnet50
```

---

### ▶️ How to Run SupCon Loss

1. Contrastive pretraining

```bash
python supcon/train_supcon.py
```

2. Linear classifier training

```bash
python supcon/train_linear.py \
  --encoder-path checkpoints_supcon/<run_timestamp>/encoder_supcon.pth \
  --split-path checkpoints_supcon/<run_timestamp>/split.json \
  --dataset-dir data/dataset
```

3. Evaluation

```bash
python supcon/eval_supcon.py \
  --encoder-path checkpoints_supcon/<run_timestamp>/encoder_supcon.pth \
  --classifier-path checkpoints_linear/<run_timestamp>/classifier.pth \
  --split-path checkpoints_supcon/<run_timestamp>/split.json \
  --dataset-dir data/dataset \
  --wandb
```

Outputs:

- `checkpoints_supcon/<time>/encoder_supcon.pth`
- `checkpoints_linear/<time>/classifier.pth`

---

### 🔄 Optional: All-in-One SupCon + Linear Evaluation (W&B Pipeline)

In addition to the separate training scripts, the repository provides a single
pipeline script that performs:

1. Supervised Contrastive (SupCon) pretraining  
2. Linear classifier training  
3. Validation and test evaluation  
4. Logging to Weights & Biases (W&B)  
5. Saving model checkpoints and split information  

---

#### ▶ Run the full pipeline (default configuration)

```bash
python supcon/<PIPELINE_SCRIPT_NAME>.py \
  --run-name-prefix <name> \
  --random-state <int> \
  --test-size <float> \
  --val-size <float> \
  --temperature <float> \
  --feature-dim <int> \
  --supcon-lr <float> \
  --supcon-epochs <int> \
  --linear-lr <float> \
  --linear-epochs <int> \
  --apply-minority-aug <true/false> \
  --minority-classes <comma-separated-class-names>
```

### How to train with synthetic images

#### 1. Generate splits with synthetic data
`json_adjuster.py` builds a series of training splits and corresponding annotations by mixing **synthetic images** into the original dataset. It:

- Copies original images and annotations into the output dataset folders.
- Samples synthetic images per class across multiple steps (see `STEPS` in the script).
- Renames synthetic images to the canonical `IMG000000.ext` format.
- Creates minimal JSON annotations for each synthetic image.
- Writes one split file per step: `split_step1.json`, `split_step2.json`, ...

Basic usage:

```bash
python json_adjuster.py \
 --input_split data/baseline_split.json \
 --output_split data/dataset/splits \
 --synthetic_images <Path to synthetic images> \
 --input_images data/dataset/final_patched_BTXRD \
 --output_images data/dataset/BTXRD_images_new \
 --input_annotations data/dataset/BTXRD/Annotations \
 --output_annotations data/dataset/BTXRD/Annotations_new
```

Key arguments:

- `--input_split`: Base split JSON (must contain a `train` list), e.g. `data/baseline_split.json`.
- `--output_split`: Output directory for incremental split files, e.g. `data/dataset/splits/` (one per step).
- `--synthetic_images`: Root folder containing class subfolders of generated images (e.g. `<Path to synthetic images>`).
- `--input_images`: Source folder with original images to copy (e.g. `data/dataset/final_patched_BTXRD`).
- `--output_images`: Target folder for originals + synthetic images (e.g. `data/dataset/BTXRD_images_new`).
- `--input_annotations`: Source folder with original JSON annotations to copy (e.g. `data/dataset/BTXRD/Annotations`).
- `--output_annotations`: Target folder for originals + synthetic JSON annotations (e.g. `data/dataset/BTXRD/Annotations_new`).

Outputs:

- `data/dataset/splits/split_step*.json` with incrementally expanded `train` indices.
- New synthetic images and JSON annotations alongside the originals.

#### 2. Train with synthetic data
You have to add trainwsyn argument to train with the specific step

Basic usage:

```bash
python train.py \
    --trainwsyn <SYNTHETIC_SPLIT> \
    --run-name-prefix <RUN_NAME_PREFIX>
```

Key arguments:
- `--trainwsyn`: Path to a synthetic split JSON (e.g. `data/dataset/splits/split_step3.json`). This selects which step’s augmented train indices are used.
- `--run-name-prefix`: Prefix for the run/checkpoint name (e.g. `resnet_gan_15800`). The final run name includes this prefix plus a timestamp/settings.

3. Test trained model


## ✨ 1.Synthetic Generation (Stylegan2)

### 1. Clone [stylegan2-ada-pytorch](https://github.com/philippw23/stylegan2-ada-pytorch)

### 2. Move data into StyleGAN repo

Place these folders under `stylegan2-ada-pytorch/data/dataset/`:

- `BTXRD/` (must contain `Annotations/`)
- `final_patched_BTXRD/` (patched images)
- `dataset_split.json` (or adapt `--split-path` below)

Expected layout:

```text
stylegan2-ada-pytorch/
  data/
    dataset/
      BTXRD/
        Annotations/
      final_patched_BTXRD/
      dataset_split.json
```

### 3. Preprocess

The full BTXRD preprocessing pipeline is handled by one script call via
`style_gan_preprocessing.py full-pipeline` (resize/sort, index-map creation,
and train-split correction).

```bash
python data/style_gan_preprocessing.py full-pipeline [arguments]
```

Arguments:

- `--image-dir` (path, default: `data/dataset/final_patched_BTXRD`): Input image directory for preprocessing.
- `--json-dir` (path, default: `data/dataset/BTXRD/Annotations`): Directory with BTXRD annotation JSON files.
- `--preprocess-output-dir` (path, default: `data/dataset/BTXRD_resized_sorted_with_anatomical_location`): Output directory for resized and class-sorted images.
- `--target-size` (int, default: `256`): Output square image size.
- `--center-crop` (flag): Center-crop images to square before resize.
- `--no-dataset-json` (flag): Skip writing `dataset.json` during preprocess step.
- `--use-anatomical-location` (flag): Prefix tumor labels with anatomical location to create 21 classes.
- `--xlsx-path` (path, default: `data/dataset/BTXRD/dataset.xlsx`): Path to metadata XLSX used when `--use-anatomical-location` is set.
- `--split-path` (path, default: `data/dataset/dataset_split.json`): Split JSON used to build index map and keep only train images.
- `--index-map-dataset-dir` (path, default: `data/dataset/final_patched_BTXRD`): Directory used to build the index-to-filename map.
- `--index-map-output-path` (path, default: `data/dataset/final_patched_index_map.json`): Output JSON path for the generated index map.
- `--correct-split-dataset-dir` (path, default: `--preprocess-output-dir`): Directory to apply train-split filtering on.
- `--dry-run` (flag): Preview split correction without deleting files or rewriting `dataset.json`.

Example:

```bash
python data/style_gan_preprocessing.py full-pipeline \
  --image-dir data/dataset/final_patched_BTXRD \
  --json-dir data/dataset/BTXRD/Annotations \
  --preprocess-output-dir data/dataset/BTXRD_resized_sorted \
  --target-size 256 \
  --split-path data/dataset/dataset_split.json \
  --index-map-dataset-dir data/dataset/final_patched_BTXRD \
  --index-map-output-path data/dataset/final_patched_index_map.json \
  --correct-split-dataset-dir data/dataset/BTXRD_resized_sorted
```

Other available subcommands:

- `preprocess`: Only resize/sort images and optionally write `dataset.json`.
- `build-index-map`: Only create `final_patched_index_map.json` from split indices.
- `correct-split`: Only filter the resized dataset to the train split and rewrite `dataset.json`.

```bash
python data/style_gan_preprocessing.py preprocess --help
python data/style_gan_preprocessing.py build-index-map --help
python data/style_gan_preprocessing.py correct-split --help
python data/style_gan_preprocessing.py full-pipeline --help
```

### 4. Pack dataset for StyleGAN2-ADA

```bash
python data/dataset_tool.py [arguments]
```

Arguments:

- `--source` (required, path): Input dataset path (folder or archive).
- `--dest` (required, path): Output dataset path (folder or archive, e.g. `.zip`).
- `--max-images` (int, optional): Limit number of images.
- `--resize-filter` (choice: `box|lanczos`, default: `lanczos`): Resize interpolation filter.
- `--transform` (choice: `center-crop|center-crop-wide`, optional): Crop/resize mode.
- `--width` (int, optional): Output width.
- `--height` (int, optional): Output height.

Example:

```bash
python data/dataset_tool.py \
  --source data/dataset/BTXRD_resized_sorted \
  --dest data/btxrd_corrected_dataset.zip \
  --width 256 --height 256 --resize-filter box
```

### 5. Train StyleGAN2-ADA

```bash
python train.py [arguments]
```

Arguments:

- `--outdir` (required): Output directory for training runs.
- `--data` (required): Training dataset path (directory or zip).
- `--gpus` (int, default: `1`): Number of GPUs (power of two).
- `--batch` (int, optional): Override batch size.
- `--gamma` (float, optional): Override R1 gamma.
- `--cond` (bool, default: `false`): Enable conditional training from labels in `dataset.json`.
- `--mirror` (bool, default: `false`): Enable horizontal flips.
- `--aug` (choice: `noaug|ada|fixed`, default: `ada`): Augmentation mode.
- `--cfg` (choice: `auto|stylegan2|paper256|paper512|paper1024|cifar`, default: `auto`): Base configuration.
- `--snap` (int, default: `50`): Snapshot interval in ticks.
- `--resume` (default: `noresume`): Resume from pickle or predefined source.
- `--kimg` (int, optional): Override training duration.
- `--seed` (int, default: `0`): Random seed.
- `--metrics` (default: `fid50k_full`): Comma-separated metric list or `none`.

Example:

```bash
python train.py \
  --outdir=/checkpoints/stylegan2ada_cond_train \
  --data=./data/btxrd_train_dataset.zip \
  --gpus=1 \
  --batch=16 \
  --gamma=6 \
  --cond=1 \
  --mirror=1 \
  --aug=ada \
  --cfg=auto \
  --snap=50 \
  --seed=42 \
  --metrics=fid50k_full
```

### 6. Generate synthetic images

After training, pick a snapshot from `training-runs/.../network-snapshot-xxxxxx.pkl` and sample images:

```bash
python generate.py [arguments]
```

Arguments:

- `--network` (required): Network pickle path (`.pkl`) or URL.
- `--outdir` (required): Output image directory.
- `--seeds` (required unless `--projected-w` is used): Comma list or range, e.g. `0,1,2` or `0-199`.
- `--trunc` (float, default: `1.0`): Truncation psi.
- `--class` (int, optional): Class ID for conditional models.
- `--noise-mode` (choice: `const|random|none`, default: `const`): Noise handling.
- `--projected-w` (file, optional): Generate from projected latent `W` instead of seeds.

Example:

```bash
python generate.py \
  --outdir out/btxrd_samples \
  --trunc 0.7 \
  --seeds 0-799 \
  --class 0 \
  --network training-runs/stylegan2ada_cond_train/00000-btxrd_train_anatomical_dataset-cond-mirror-auto1-gamma6-batch16-ada/network-snapshot-020000.pkl
```

Notes:

- `--class` is required when training with `--cond true`; class IDs come from `dataset.json`.
- Repeat with different `--class` values to generate each tumor class or use the generate.sbatch file
- For each class a minimum of 800 images should be generated for the execution of json-adjuster to    work.

---

## ✨ 2.Synthetic Generation (Custom Latent Diffusion)

### Autoencoder

#### 1. Train the autoencoder

```
python -m custom_latent_diffusion.vae.train
```

#### 2. Test the autoencoder (Optional)

```
python -m custom_latent_diffusion.vae.train --run-name <RUN_NAME>
```

#### 3. Sample from the autoencoder (Optional)

```
python -m custom_latent_diffusion.vae.sample --run-name <RUN_NAME>
```

---

**`<RUN_DIR>` is the directory of the run which you want to test, for example `train_vae_2025-12-07_17-36-29`**

### Diffusion Model

#### Train the diffusion model using a latent space provided by a VAE.

```
python -m custom_latent_diffusion.diffusion.train --run-name <RUN_NAME>
```

**`<RUN_DIR>` is the directory of the VAE train run, for example `train_vae_2025-12-07_17-36-29`**

### Sample

```
python -m custom_latent_diffusion.sample --vae-run-name <VAE_RUN_NAME> --ldm-run-name <LDM_RUN_NAME> --class-name <CLASS_NAME>
```

- **`<VAE_RUN_DIR>` is the directory of the VAE train run, for example `train_vae_2025-12-07_17-36-29`**
- **`<LDM_RUN_DIR>` is the directory of the diffusion train run, for example `train_ldm_2025-12-07_17-36-29`**
- **`<CLASS_NAME>` is the name of the tumor subtype which you wish to sample for, for example `osteochondroma`**

## ✨ 3.Synthetic Generation (New Custom Latent Diffusion)

Work had been started on a new custom latent diffusion approach that uses the diffusers library in the `custom_latent_diffusion_new` folder. However, this remains work in progress and is not usable yet.

## 📏 Metric Evaluation for Image Generation

This repo provides two complementary metrics to assess generated image quality and diversity:

- **LPIPS (intra-class diversity):** `lpips_eval.py`
- **FID (distribution distance real vs. generated):** `finetuned_latent_diffusion/evaluate_fid.sh`

### 1. LPIPS evaluation (`lpips_eval.py`)

Run from project root:

```bash
python lpips_eval.py \
  --real_root data/dataset/final_patched_BTXRD \
  --gen_root <GENERATED_ROOT> \
  --pairs 10000 \
  --img_size 256 \
  --backbone alex
```

Arguments:

- `--real_root` (required): Root directory with real class subfolders.
- `--gen_root` (required): Root directory with generated class subfolders.
- `--classes` (optional): Explicit class list. If omitted, all classes under `real_root` are used.
- `--pairs` (int, default: `10000`): Number of random image pairs per class.
- `--img_size` (int, default: `256`): Resize target for LPIPS input.
- `--backbone` (choice: `alex|vgg|squeeze`, default: `alex`): LPIPS backbone.
- `--device` (default: auto): Device for evaluation (`cuda` if available, else `cpu`).
- `--seed` (int, default: `0`): Random seed for pair sampling.

Output:

- Per-class LPIPS mean/std for real and generated sets.
- Macro average across classes.

### 2. FID evaluation (`finetuned_latent_diffusion/evaluate_fid.sh`)

The SLURM script computes FID with `pytorch-fid`:

```bash
sbatch finetuned_latent_diffusion/evaluate_fid.sh
```

Before running, edit these variables in `finetuned_latent_diffusion/evaluate_fid.sh`:

- `REAL_DIR`: Flattened folder of real images.
- `FAKE_DIR`: Flattened folder of generated images to evaluate.
