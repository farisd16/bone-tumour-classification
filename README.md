# 🦴 **ADLM: Addressing Class Imbalance in Bone Tumor X-Ray Classification with Generative Models**

This project explores how **generative models** can help address **class imbalance** in **bone tumor X-ray classification**.  
We use the **BTXRD dataset**, which contains X-ray images of different primary bone tumor entities.

---

## **🎯 Project Goals**

- Implement a **ResNet-based CNN** as a **baseline model** for tumor classification.  
- Apply and analyze **class imbalance handling techniques** such as **Weighted Loss**, **Focal Loss**, and **Data Augmentation**.  
- Use **generative models** (e.g., **GANs** or **Diffusion Models**) to create synthetic X-ray images and evaluate their impact on classification performance.  

---

## **🦴 Target Tumor Types**

We focus on classifying **seven tumor types**:

- Osteochondroma  
- Osteosarcoma  
- Multiple Osteochondromas  
- Simple Bone Cyst  
- Giant Cell Tumor  
- Synovial Osteochondroma  
- Osteofibroma  

---

## **🏫 About the Project**

This project is part of the **Advanced Deep Learning Methods (ADLM)** course at the  
**Technical University of Munich (TUM)**, in collaboration with the  
**Clinic for Orthopaedics and Sports Orthopaedics** and the  
**Institute for AI and Informatics in Medicine**.

---

## **⬇️ Dataset Setup**

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

## **▶️ How To Run (Preparation → Training → Testing)**

1) Extract patches from annotations

```
python data/btxrd_bounding_box_dataset_extractor.py
```

This creates `data/dataset/final_patched_BTXRD/` from `BTXRD/images` + `BTXRD/Annotations`.

2) Train (with optional early stopping)

```
python src/training_ResNet.py --model resnet50 --early-stop --patience 10 --min-delta 0.001
```

Notes:
- The training pipeline reads labels directly from JSON annotations and splits in-memory.
- By default it uses `data/dataset/patched_BTXRD_merged/` if present, otherwise falls back to `patched_BTXRD/`.
- Checkpoints are written under `checkpoints/<model>/`.

3) Test and generate confusion matrix

```
python src/testing_ResNet.py --model resnet50
```

Outputs:
- `checkpoints/<model>/test_predictions.npy`
- `checkpoints/<model>/confusion_matrix.png`

4) (Optional) Quick visualization of predictions

```
python src/plot_predictions.py
```

---
## **▶️ How to Run SupCon Loss **

1) Contrastive Pretraining (train_supcon.py)
```
python train_supcon.py
```
2) Linear Classifier Training (train_linear.py)
```
python train_linear.py
```
3) Evaluation (eval_supcon.py)
```
python eval_supcon.py
```

Outputs:
- `checkpoints_supcon/<time>/encoder_supcon.pth`
- `checkpoints_linear/<time>/classifier.pth`

## **ℹ️ Notes**

- CSVs like `dataset_singlelabel.csv` are not required for training/testing in this pipeline; labels are taken from annotation JSONs. If needed for analysis, you can generate a CSV aligned to the patched images via:

```
python data/create_csv_patched.py
```
