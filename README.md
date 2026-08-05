# Deep Learning Portfolio

Six from-scratch deep learning projects, covering the core model families: regression, contrastive/metric learning, sequence generation (captioning and QA), object detection, and generative modeling (diffusion).

| Project | What it does | Result |
|---|---|---|
| [`LinearRegression`](LinearRegression) | Neural-net linear regression on California housing prices, with a manual hyperparameter sweep and a debugged exploding-gradients failure case. | Best val loss 0.261 at lr=0.01, hidden=8 |
| [`ImageRetrievalWithCL`](ImageRetrievalWithCL) | Siamese network + contrastive loss for image similarity on Caltech-101, with tuned similarity threshold. | 68.6% test accuracy, 72.3% F1 |
| [`CaptionGeneration`](CaptionGeneration) | ResNet50 encoder + LSTM decoder image captioning. | BLEU ~0.1 after 10 epochs, no overfitting |
| [`ObjectDetectionYOLO`](ObjectDetectionYOLO) | YOLOv8-nano vs YOLOv8-small compared on a custom vehicle-detection dataset. | Nano: 0.976 mAP@0.5, beating the larger model |
| [`TransformerQA`](TransformerQA) | From-scratch GPT-style decoder-only Transformer for generative question answering. | Early-stopped, BLEU/perplexity tracked per epoch |
| [`ImageGenerationDiffusion`](ImageGenerationDiffusion) | DDPM-style diffusion model trained from scratch on a 5-class animal image set. | Denoises pure noise back to a recognizable image |

Each project is self-contained with its own `README.md` and `requirements.txt`. None of the projects share code — they're independent implementations built to understand each model family from first principles rather than calling a high-level library.

## Running any project

```bash
cd <project-folder>
pip install -r requirements.txt
```

Datasets are **not** included in this repo (they're standard public datasets — California Housing, Caltech-101, a Flickr8k-style captioning set, a Stack Overflow Q&A dump, Animals-10). Each project's README says exactly which dataset it needs and what folder layout to place it in before running.

## What's genuinely "from scratch" vs. using a library

- `LinearRegression`, `ImageRetrievalWithCL`, `CaptionGeneration`, `TransformerQA`, `ImageGenerationDiffusion` — model architecture, training loop, and loss functions are hand-written in PyTorch/TensorFlow.
- `ObjectDetectionYOLO` — uses the `ultralytics` YOLOv8 implementation directly (the contribution here is the dataset pipeline and the nano-vs-small comparison, not the detector architecture itself).
