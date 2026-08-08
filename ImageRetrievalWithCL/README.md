# Contrastive Learning for Image Similarity

A Siamese network trained with contrastive loss to decide whether two images belong to the same class — the model never sees class labels directly during the similarity task, only pairs of images and a same/different signal.

## Architecture

Both branches of the Siamese network share a single ImageNet-pretrained **ResNet50** (`torchvision.models.resnet50(pretrained=True)`) as the backbone, with its final classification layer replaced by a custom projection head (2048 → 512 → 128) producing an L2-normalized embedding. Similarity between two images is the Euclidean distance between their embeddings.

## Approach

1. **Balanced pair sampling** — the data loader constructs balanced same-class/different-class image pairs from Caltech-101 so the model can't shortcut by predicting the majority class.
2. **Two-stage training**: first trained for 10 epochs at a constant LR to get a baseline embedding space, then used the validation set to tune the similarity threshold (the Euclidean-distance cutoff that separates "same" from "different").
3. **Threshold tuning** — swept thresholds from 0.50 to 0.62; picked **0.52** as the best accuracy/F1 tradeoff (see table below), then retrained for 15 epochs with a decaying LR schedule using that fixed threshold.

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.50 | 68.4% | 71.2% | 66.9% | 69.0% |
| **0.52** | **69.1%** | **70.4%** | **70.8%** | **70.6%** |
| 0.55 | 69.9% | 69.1% | 77.3% | 73.0% |
| 0.62 | 69.5% | 65.6% | 88.1% | 75.2% |

## Final test results

```
Test Loss: 0.1934
Accuracy: 68.6%   Precision: 65.5%   Recall: 80.6%   F1: 72.3%
```

Training logs (reflected in `graphs/loss.png`) show the model stuck in a local minimum around epoch 4 — loss stopped decreasing until the LR was manually dropped from 0.01 to 0.005, after which it resumed improving.

## Dataset

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) — not included in this repo. Download and place it here:

```
ImageRetrievalWithCL/
  dataset/
    caltech-101/
      <class_name>/
        image_XXXX.jpg
      ...
```

## Setup

```bash
pip install -r requirements.txt
python train.py   # trains and saves weights.pth
python test.py    # evaluates on the held-out test split
python test_eval.py  # quick sanity check on a hardcoded pair of images from the dataset
```

## Project structure

```
model.py        # SiameseNetwork architecture
data_utils.py   # pair sampling, dataset loading
utils.py        # contrastive loss, metrics, threshold-based prediction
train.py        # training loop
test.py          # test-set evaluation
test_eval.py     # single-pair inference example
graphs/          # accuracy/precision/recall/F1/loss curves across training
```

## Stack

PyTorch, torchvision (ResNet50 backbone, ImageNet-pretrained), scikit-learn (metrics).
