# Research Lab Assignment

This repository contains implementations for two computer vision tasks: Vision Transformer classification and text-driven segmentation.

---

## Q1: Vision Transformer on CIFAR-10

Implementation of Vision Transformer (ViT) for CIFAR-10 classification achieving **84.22% test accuracy**.

### How to Run

1. Open `q1.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. The notebook automatically downloads CIFAR-10 and trains the model

### Pre-trained Model

Download the trained model weights:
- **GitHub Release**: [Download vit_cifar10_best.pth](https://github.com/gadhalekshmip/yourrepo/releases/download/v1.0/vit_cifar10_best.pth) (57MB)
- **Google Drive**: [Download Link](https://drive.google.com/file/d/1Al61Qw3zq3xRy116OCC6j4QWUG2TQXFG/view?usp=sharing)

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 32×32 |
| Patch Size | 4×4 |
| Embedding Dimension | 384 |
| Transformer Blocks | 8 |
| Attention Heads | 6 |
| MLP Ratio | 4 |
| Dropout | 0.1 |
| Total Parameters | 14.2M |

### Training Setup

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.05)
- **Scheduler**: Cosine annealing with warmup
- **Batch Size**: 128
- **Epochs**: 109 (early stopped)
- **Augmentation**: RandomCrop, HorizontalFlip, ColorJitter, RandomRotation, RandomErasing
- **Label Smoothing**: 0.1

### Results

**Overall Test Accuracy: 84.22%**

| Class | Accuracy |
|-------|----------|
| Airplane | 85.7% |
| Automobile | 92.2% |
| Bird | 79.7% |
| Cat | 73.1% |
| Deer | 81.6% |
| Dog | 76.0% |
| Frog | 87.1% |
| Horse | 89.0% |
| Ship | 88.9% |
| Truck | 88.9% |

### Analysis

**Patch Size**: 4×4 patches optimal for CIFAR-10's 32×32 resolution. Larger patches (8×8, 16×16) discard too much spatial detail.

**Overfitting**: 10% train-test gap (94% train vs 84% test) indicates overfitting, common for ViT on small datasets. Heavy augmentation was necessary to reach 84%.

**Architecture Choices**: 8-layer transformer with 384-dim embeddings balanced capacity and overfitting. Deeper models (12+ layers) overfit more without improving test accuracy.

**Limitations**: CIFAR-10's 50k images is small for transformers, which typically require large datasets. Pre-training on ImageNet would likely improve results significantly.

---

## Q2: Text-Driven Image Segmentation with SAM 2

Zero-shot object segmentation using natural language prompts via CLIPSeg and SAM 2.

### How to Run

1. Open `q2.ipynb` in Google Colab
2. Set runtime to GPU
3. Run all cells - models download automatically
4. Use provided examples or upload your own images

### Pipeline

```
Text Prompt + Image
    ↓
CLIPSeg (text-to-region grounding)
    ↓
Bounding Boxes
    ↓
SAM 2 (precise segmentation)
    ↓
Segmented Mask Overlay
```

### Usage

```python
# Basic usage
image = download_image("image_url")
masks, boxes = text_driven_segmentation(image, "dog")

# Adjust detection threshold
masks, boxes = text_driven_segmentation(image, "cat", threshold=0.3)
```

### Models Used

- **CLIPSeg** (CIDAS/clipseg-rd64-refined): Vision-language model for text-to-region localization
- **SAM 2** (Hiera-Large): Promptable segmentation from Meta AI


### Technical Details

- CLIPSeg provides coarse localization via attention heatmaps
- Contour detection on thresholded heatmaps generates bounding boxes
- SAM 2 refines boxes to pixel-accurate masks
- Visualization shows 4-panel output: original, heatmap, boxes, final mask

---

## Repository Structure

```
├── q1.ipynb          # Vision Transformer implementation
├── q2.ipynb          # Text-driven segmentation
└── README.md         # This file
```

