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
**Technical University of Munich (TUM)**, in collaboration with the  
**Clinic for Orthopaedics and Sports Orthopaedics** and the  
**Institute for AI and Informatics in Medicine**.

## ⬇️ Dataset Setup

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

Install dependencies:

```
pip install -r requirements.txt
```

---

## 📊 Classification

### ▶️ How To Run (Preparation → Training → Testing)

1. Extract patches from annotations

```
python data/btxrd_bounding_box_dataset_extractor.py
```

This creates `data/dataset/final_patched_BTXRD/` from `BTXRD/images` + `BTXRD/Annotations`.

2. Train (with optional early stopping)

```
python src/training_ResNet.py --model resnet50 --early-stop --patience 10 --min-delta 0.001
```

Notes:

- The training pipeline reads labels directly from JSON annotations and splits in-memory.
- By default it uses `data/dataset/patched_BTXRD_merged/` if present, otherwise falls back to `patched_BTXRD/`.
- Checkpoints are written under `checkpoints/<model>/`.

3. Test and generate confusion matrix

```
python src/testing_ResNet.py --model resnet50
```

Outputs:

- `checkpoints/<model>/test_predictions.npy`
- `checkpoints/<model>/confusion_matrix.png`

4. (Optional) Quick visualization of predictions

```
python src/plot_predictions.py
```

---

### ▶️ How to Run SupCon Loss

1. Contrastive Pretraining (run train_supcon.py)

2. Linear Classifier Training (run train_linear.py)

3. Evaluation (run eval_supcon.py)

Outputs:

- `checkpoints_supcon/<time>/encoder_supcon.pth`
- `checkpoints_linear/<time>/classifier.pth`

### ℹ️ Notes

- CSVs like `dataset_singlelabel.csv` are not required for training/testing in this pipeline; labels are taken from annotation JSONs. If needed for analysis, you can generate a CSV aligned to the patched images via:

```
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

Philipp