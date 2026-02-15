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
conda create -n bone-tumour python=3.12
conda activate bone-tumour
```

**Option 2: Using venv**

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

Optional folders created by scripts in this repo:

```
data/
  dataset/
    final_patched_BTXRD/          # Extracted patches from annotations (created, this is the final dataset used for training and testing)
    squared_padded/         # Padded originals for 106 special cases (created)
    squared_patched_106/    # Patches from padded images (created)
    patched_BTXRD_merged/   # Merge of the two patch sets (created)
```

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

Arguments:

- No CLI arguments are currently implemented in `supcon/train_supcon.py`.

Example:

```bash
python supcon/train_supcon.py
```

2. Linear classifier training

```bash
python supcon/train_linear.py
```

Arguments:

- No CLI arguments are currently implemented in `supcon/train_linear.py`.

Example:

```bash
python supcon/train_linear.py
```

3. Evaluation

```bash
python supcon/eval_supcon.py
```

Arguments:

- No CLI arguments are currently implemented in `supcon/eval_supcon.py`.

Example:

```bash
python supcon/eval_supcon.py
```

Outputs:

- `checkpoints_supcon/<time>/encoder_supcon.pth`
- `checkpoints_linear/<time>/classifier.pth`

### ℹ️ Notes

- CSVs like `dataset_singlelabel.csv` are not required for training/testing in this pipeline; labels are taken from annotation JSONs.
- If needed for analysis, you can generate a CSV aligned to patched images via:

```bash
python data/create_csv_patched.py
```

---

## 🆕 1.Synthetic Generation (Latent Diffusion)

### Autoencoder

#### 1. Train the autoencoder

```
python -m latent_diffusion.vae.train
```

#### 2. Test the autoencoder (Optional)

```
python -m latent_diffusion.vae.train --run-name <RUN_NAME>
```

#### 3. Sample from the autoencoder (Optional)

```
python -m latent_diffusion.vae.sample --run-name <RUN_NAME>
```

---

**`<RUN_DIR>` is the directory of the run which you want to test, for example `train_vae_2025-12-07_17-36-29`**

### Diffusion Model

#### Train the diffusion model using a latent space provided by a VAE.

```
python -m latent_diffusion.diffusion.train --run-name <RUN_NAME>
```

**`<RUN_DIR>` is the directory of the VAE train run, for example `train_vae_2025-12-07_17-36-29`**

### Sample

```
python -m latent_diffusion.sample --vae-run-name <VAE_RUN_NAME> --ldm-run-name <LDM_RUN_NAME> --class-name <CLASS_NAME>
```

- **`<VAE_RUN_DIR>` is the directory of the VAE train run, for example `train_vae_2025-12-07_17-36-29`**
- **`<LDM_RUN_DIR>` is the directory of the diffusion train run, for example `train_ldm_2025-12-07_17-36-29`**
- **`<CLASS_NAME>` is the name of the tumor subtype which you wish to sample for, for example `osteochondroma`**

## 🆕 2.Synthetic Generation (Stylegan2)

Run the following from `stylegan2-ada-pytorch/` (not from this repo root).

### 1. Move data into StyleGAN repo

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

### 2. Preprocess and create class-sorted 256x256 dataset

```bash
python data/style_gan_preprocessing.py [arguments]
```

Arguments:

- `--image-dir` (path, default: `data/dataset/final_patched_BTXRD`): Input image directory.
- `--json-dir` (path, default: `data/dataset/BTXRD/Annotations`): JSON annotation directory.
- `--output-dir` (path, default: `data/dataset/BTXRD_resized_sorted_with_anatomical_location`): Output directory.
- `--target-size` (int, default: `256`): Output square size.
- `--center-crop` (flag): Center-crop to square before resize.
- `--no-dataset-json` (flag): Skip writing `dataset.json`.
- `--use-anatomical-location` (flag): Prefix class label with anatomical location (`upper limb`, `lower limb`, `pelvis`) to create 21 classes.
- `--xlsx-path` (path, default: `data/dataset/BTXRD/dataset.xlsx`): Metadata file used with `--use-anatomical-location`.

Example:

```bash
python data/style_gan_preprocessing.py \
  --image-dir data/dataset/final_patched_BTXRD \
  --json-dir data/dataset/BTXRD/Annotations \
  --output-dir data/dataset/BTXRD_resized_sorted \
  --target-size 256
```

Example (21 classes with anatomical prefix):

```bash
python data/style_gan_preprocessing.py \
  --image-dir data/dataset/final_patched_BTXRD \
  --json-dir data/dataset/BTXRD/Annotations \
  --output-dir data/dataset/BTXRD_resized_sorted_with_anatomical_location \
  --target-size 256 \
  --use-anatomical-location
```

### 3. Build index-to-filename map from original patched dataset

```bash
python data/build_final_patched_index_map.py [arguments]
```

Arguments:

- `--split-path` (path, default: `data/dataset/dataset_split.json`): Split JSON containing `train` and `test` indices.
- `--dataset-dir` (path, default: `data/dataset/final_patched_BTXRD`): Directory with `.jpeg` files in the original ordering.
- `--output-path` (path, default: `data/dataset/final_patched_index_map.json`): Output index-to-filename map.

Example:

```bash
python data/build_final_patched_index_map.py \
  --split-path data/dataset/dataset_split.json \
  --dataset-dir data/dataset/final_patched_BTXRD \
  --output-path data/dataset/final_patched_index_map.json
```

`build_final_patched_index_map.py` is needed because split files contain integer indices, not filenames.  
It creates `index -> IMGxxxx.jpeg` mapping using the original `final_patched_BTXRD` ordering so split indices can be matched to resized/sorted files.

### 4. Keep only train split in the resized dataset

```bash
python data/correct_split_new.py [arguments]
```

Arguments:

- `--split-path` (path, default: `data/dataset/dataset_split.json`): Split JSON (uses only `train` indices).
- `--dataset-dir` (path, default: `data/dataset/BTXRD_resized_sorted`): Resized class-sorted dataset directory.
- `--index-map` (path, default: `data/dataset/final_patched_index_map.json`): Index-to-filename mapping JSON.
- `--dry-run` (flag): Preview removals without deleting files or rewriting `dataset.json`.

Example:

```bash
python data/correct_split_new.py \
  --split-path data/dataset/dataset_split.json \
  --dataset-dir data/dataset/BTXRD_resized_sorted \
  --index-map data/dataset/final_patched_index_map.json \
  --dry-run
```

### 5. Pack dataset for StyleGAN2-ADA

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

### 6. Train StyleGAN2-ADA

```bash
python train.py [arguments]
```

Arguments:

- `--outdir` (required): Output directory for training runs.
- `--data` (required): Training dataset path (directory or zip).
- `--gpus` (int, default: `1`): Number of GPUs (power of two).
- `--snap` (int, default: `50`): Snapshot interval in ticks.
- `--metrics` (default: `fid50k_full`): Comma-separated metric list or `none`.
- `--seed` (int, default: `0`): Random seed.
- `-n`, `--dry-run` (flag): Print config and exit without training.
- `--cond` (bool, default: `false`): Enable conditional training from labels in `dataset.json`.
- `--subset` (int, optional): Train on only N images.
- `--mirror` (bool, default: `false`): Enable horizontal flips.
- `--cfg` (choice: `auto|stylegan2|paper256|paper512|paper1024|cifar`, default: `auto`): Base configuration.
- `--gamma` (float, optional): Override R1 gamma.
- `--kimg` (int, optional): Override training duration.
- `--batch` (int, optional): Override batch size.
- `--aug` (choice: `noaug|ada|fixed`, default: `ada`): Augmentation mode.
- `--p` (float, optional): Augmentation probability, only for `--aug=fixed`.
- `--target` (float, optional): ADA target, only for `--aug=ada`.
- `--augpipe` (choice: `blit|geom|color|filter|noise|cutout|bg|bgc|bgcf|bgcfn|bgcfnc`, default: `bgc`): Augmentation pipeline.
- `--resume` (default: `noresume`): Resume from pickle or predefined source.
- `--freezed` (int, default: `0`): Number of frozen discriminator layers.
- `--fp32` (bool, default: `false`): Disable mixed precision.
- `--nhwc` (bool, default: `false`): Use NHWC layout with FP16.
- `--nobench` (bool, default: `false`): Disable cuDNN benchmarking.
- `--allow-tf32` (bool, default: `false`): Allow TF32 in PyTorch ops.
- `--workers` (int, default: `3`): DataLoader worker count.

Example:

```bash
python train.py \
  --outdir training-runs \
  --data data/btxrd_corrected_dataset.zip \
  --gpus 1 \
  --cfg auto \
  --cond true \
  --snap 10
```

### 7. Generate synthetic images

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
  --seeds 0-199 \
  --class 0 \
  --network training-runs/<RUN_DIR>/network-snapshot-020000.pkl
```

Notes:

- `--class` is required when training with `--cond true`; class IDs come from `dataset.json`.
- Repeat with different `--class` values to generate each tumor class.

---

## 📏 Metric Evaluation for Image Generation

This repo provides two complementary metrics to assess generated image quality and diversity:

- **LPIPS (intra-class diversity):** `lpips_eval.py`
- **FID (distribution distance real vs. generated):** `latent_diffusion_finetuned/evaluate_fid.sh`

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

### 2. FID evaluation (`latent_diffusion_finetuned/evaluate_fid.sh`)

The SLURM script computes FID with `pytorch-fid`:

```bash
sbatch latent_diffusion_finetuned/evaluate_fid.sh
```

Before running, edit these variables in `latent_diffusion_finetuned/evaluate_fid.sh`:

- `REAL_DIR`: Flattened folder of real images.
- `FAKE_DIR`: Flattened folder of generated images to evaluate.
