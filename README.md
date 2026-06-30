# 🫁 COVID-19 Chest X-Ray Classification

A deep learning pipeline for classifying chest X-ray images into **COVID-19**, **Normal**, and **Viral Pneumonia**, built with PyTorch and benchmarking four state-of-the-art CNN architectures. The project goes beyond plain classification by adding an **out-of-distribution (OOD) rejection mechanism** ("UNKNOWN" class via a dynamic confidence threshold), **Grad-CAM explainability**, and a **t-SNE analysis of the learned latent space**.

---

## Overview

Given a chest X-ray, the model predicts one of three classes — COVID-19, Normal, or Viral Pneumonia — and is able to flag low-confidence predictions as `UNKNOWN` rather than forcing a guess. Four backbones are trained and compared under an identical, heavily optimized training pipeline, and the best-performing model is further analyzed with explainability and error-analysis tools.

## Key Features

- **Multi-architecture benchmark**: EfficientNet-B3, DenseNet-121, ResNet-50, ConvNeXt-Small, all initialized with ImageNet weights and fine-tuned with a custom classification head.
- **Optimized training loop**: TF32 matmuls, cuDNN autotuning, mixed-precision (AMP) training, `torch.compile` graph compilation, gradient clipping, label smoothing, AdamW + cosine annealing, and early stopping.
- **High-throughput data pipeline**: images are cached in RAM as raw bytes at dataset construction time, removing disk I/O from the training loop; `DataLoader`s use persistent workers, prefetching, and pinned memory.
- **Dynamic confidence threshold (OOD detection)**: predictions are rejected as `UNKNOWN` when the maximum softmax probability falls below `mean − k·std`, with a full sensitivity analysis over `k`.
- **Error analysis**: pixel intensity, brightness, and contrast distributions are compared between accepted and rejected (`UNKNOWN`) images to understand *why* the model is uncertain.
- **Explainability**: Grad-CAM heatmaps on both the most confused and the most confidently-correct predictions, plus a t-SNE projection of the 512-dimensional penultimate-layer features.
- **Reproducible EDA**: class balance, pixel intensity histograms, sample grids, and per-class "average X-ray" visualizations.

## Dataset

The project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle), which contains chest X-ray images across four original categories (COVID, Normal, Viral Pneumonia, Lung Opacity — only the first three are used here).

| Class | Original count | Balanced (used) |
|---|---|---|
| Normal | 10,192 | 1,345 |
| COVID | 3,616 | 1,345 |
| Viral Pneumonia | 1,345 | 1,345 |

The dataset is undersampled to the minority class (Viral Pneumonia) to avoid class imbalance, then split **60% / 20% / 20%** into stratified train / validation / test sets (2,421 / 807 / 807 images).

> ⚠️ The dataset is **not included** in this repository due to its size and license. Download it from Kaggle and update the `PATHS` dictionary in the notebook accordingly.

## Methodology

1. **Exploratory Data Analysis** — class distribution before/after balancing, pixel intensity histograms per class, sample image grids, and per-class average X-ray.
2. **Preprocessing & Augmentation** — resize to 224×224, random crop, horizontal flip, random rotation, color jitter (training only); ImageNet normalization for all splits.
3. **Model Training** — each of the four backbones is fine-tuned for up to 10 epochs with AdamW, cosine annealing LR, label smoothing, gradient clipping, and early stopping (patience = 3).
4. **Evaluation** — accuracy, macro precision/recall/F1, per-class classification report, and confusion matrices on the held-out test set.
5. **OOD / UNKNOWN Detection** — a dynamic threshold `mean(max_prob) − k·std(max_prob)` rejects low-confidence predictions; `k` is tuned and its sensitivity analyzed for each model.
6. **Error Analysis** — comparison of pixel-level statistics (intensity, brightness, contrast) between known (accepted) and unknown (rejected) test images.
7. **Explainability** — Grad-CAM on the best model's most-confused and most-confident predictions; t-SNE visualization of the learned feature space, colored by class and known/unknown status.

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNet-B3 | 0.9616 | 0.9616 | 0.9616 | 0.9616 |
| DenseNet-121 | 0.9492 | 0.9493 | 0.9492 | 0.9489 |
| **ResNet-50** | **0.9715** | **0.9724** | **0.9715** | **0.9716** |
| ConvNeXt-Small | 0.9504 | 0.9518 | 0.9504 | 0.9501 |

**ResNet-50** achieved the best overall performance and was selected for the downstream error analysis, Grad-CAM, and t-SNE sections.

### UNKNOWN-class statistics (k = 0.5)

| Model | Mean confidence | Std | Threshold | UNKNOWN (of 807) | Accuracy on accepted |
|---|---|---|---|---|---|
| EfficientNet-B3 | 0.8896 | 0.1017 | 0.8389 | 139 | 0.9940 |
| DenseNet-121 | 0.9033 | 0.0914 | 0.8574 | 112 | 0.9827 |
| ResNet-50 | 0.9131 | 0.0768 | 0.8745 | 95 | 0.9916 |
| ConvNeXt-Small | 0.9136 | 0.0984 | 0.8643 | 129 | 0.9853 |

Rejecting low-confidence predictions consistently pushes accuracy on the remaining (accepted) samples above **98–99%**, at the cost of leaving 12–17% of the test set unclassified.

## Project Structure

```
.
├── COVID19_Xray_chest_Classification.ipynb   # main notebook (full pipeline)
├── README.md
└── outputs/                                   # generated on run: plots, confusion matrices, Grad-CAM, t-SNE
```

## Getting Started

### Requirements

- Python 3.10+
- A CUDA-capable GPU (the notebook was developed and benchmarked on a Tesla T4, 16 GB VRAM)
- PyTorch 2.x (for `torch.compile` support)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow opencv-python pytorch-grad-cam
```

### Usage

1. Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and extract it locally.
2. Update the `BASE` path in the notebook's setup cell to point to your local dataset folder.
3. Run the notebook top to bottom: `jupyter notebook COVID19_Xray_chest_Classification.ipynb`.
4. To skip training, point `WEIGHTS_DIR` to a folder containing pretrained `.pth` checkpoints (one per architecture) and run only the evaluation / analysis sections.

## Tech Stack

`PyTorch` · `torchvision` · `scikit-learn` · `pandas` / `NumPy` · `Matplotlib` / `Seaborn` · `OpenCV` · `pytorch-grad-cam` · `t-SNE (scikit-learn)`

## Acknowledgments

- Dataset: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) by Tawsifur Rahman et al.
- Pretrained backbones from `torchvision.models` (ImageNet weights).
- [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) for explainability visualizations.

## License

This project is released under the MIT License. The underlying X-ray dataset retains its own license — please refer to the original Kaggle source for terms of use.

## Disclaimer

This project is for **research and educational purposes only**. It is not a certified diagnostic tool and must not be used for actual clinical decision-making.
