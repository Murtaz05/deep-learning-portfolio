# Question Answering with a Custom Transformer

A generative QA system: given a question's title and description, a from-scratch decoder-only Transformer (GPT-style) generates a natural-language answer. Built to understand transformer internals rather than fine-tuning a pretrained LM.

## Model

- Custom `MiniTransformer`: 2 decoder layers, `d_model=256`, 8 attention heads, learned token embeddings + positional encodings, causal self-attention, linear output projection.
- Tokenization via HuggingFace's `AutoTokenizer` (GPT tokenizer); padding tokens masked with `-100` so they don't contribute to the loss.
- Title + description concatenated as the input; the answer is the training target.

## Training

- CrossEntropyLoss (padding ignored), Adam optimizer, CosineAnnealingLR schedule.
- Early stopping after 3 epochs with no validation-loss improvement.
- Tracked per epoch: train/val loss, perplexity, BLEU score.
- Generation uses top-k/top-p sampling (`test.py`).

## Observations

- Validation loss decreased steadily and consistently — no signs of instability once padding was handled correctly (this was the trickiest part to get right).
- Ran into CUDA out-of-memory errors initially; fixed by lowering batch size.
- More training data would likely improve results further — BLEU/perplexity were tracked but the dataset used is relatively small for a generative task.

## Dataset

A Stack Overflow-style Q&A dataset (`Questions.csv`, `Answers.csv`, `Tags.csv`) — e.g. the [Stack Overflow "10% sample" dataset on Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions). Not included in this repo. Place it here:

```
TransformerQA/
  dataset/
    Questions.csv
    Answers.csv
    Tags.csv
```

## Setup

```bash
pip install -r requirements.txt
python train.py   # trains, tracks loss/perplexity/BLEU, saves the best checkpoint
python test.py    # interactive: type a question, get a generated answer
```

## Project structure

```
model.py                 # MiniTransformer: decoder-only architecture
data_utils.py              # dataset loading, tokenization, padding/masking
train.py                    # training loop: early stopping, LR scheduling, BLEU tracking
test.py                      # interactive top-k/top-p answer generation
msds24040_04_task1.py         # standalone script: full pipeline in one file
main.ipynb                     # notebook version of the same pipeline
```

## Stack

PyTorch, HuggingFace `transformers` (tokenizer only — the model itself is custom), NLTK (BLEU scoring).
