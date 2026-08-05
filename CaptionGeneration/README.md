# Image Captioning (ResNet50 + LSTM)

An encoder-decoder image captioning model: a ResNet50 CNN (pretrained on ImageNet) extracts visual features, projected into an embedding space, and an LSTM decoder generates a caption word-by-word conditioned on those features.

## Architecture

- **Encoder**: ResNet50 → dense projection layer into a smaller embedding space.
- **Decoder**: LSTM that takes the image embedding plus the word embeddings of the caption-so-far and predicts the next word.
- **Training**: SparseCategoricalCrossentropy loss, Adam (lr=0.001), 10 epochs, batch size 32.

## Results

- Training loss: ~3.0 → ~1.6 over 10 epochs; validation loss: ~2.5 → ~1.5. The smooth, matched decline indicates the model converged without overfitting.
- BLEU score improved from ~0 (random init) to ~0.1 — modest, but reasonable for only 10 epochs without beam search or attention.

## Dataset

Flickr8k-style captioning dataset (`captions.txt` + an `Images/` folder) — not included in this repo. Place it here:

```
CaptionGeneration/
  dataset/
    captions.txt
    Images/
      <image files>
```

`tokenizer.pkl` (the fitted vocabulary) is included in this repo. Trained weights (`best_model.weights.h5`) are **not** included — run `train.py` to produce them before running `test.py`/`test_eval.py`.

## Setup

```bash
pip install -r requirements.txt
python train.py       # trains and saves best_model.weights.h5 + tokenizer.pkl
python test.py        # generates captions on the held-out test split
python test_eval.py   # generates + visualizes a caption for one example image
```

## Project structure

```
model.py             # encoder-decoder architecture
data_utils.py         # caption/image dataset loading, preprocessing
imports.py             # shared imports used across the other files
train.py                # training loop
test.py                 # test-split caption generation
test_eval.py            # single-image caption + visualization
msds24040_03_task1.py   # standalone script: full pipeline in one file (data -> train -> eval)
task1.ipynb              # notebook version of the same pipeline
tokenizer.pkl             # fitted vocabulary (included)
```

## Stack

TensorFlow/Keras (ResNet50, LSTM), NLTK (BLEU scoring).
