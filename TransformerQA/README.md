
# Question Answering with MiniTransformer

This project implements a generative Question Answering system using a custom Transformer model trained on question-answer pairs. The model is based on a simplified GPT-style decoder architecture and is capable of generating natural language answers from input questions using top-k/top-p decoding.

---

## 📁 Project Structure

```
project/
├── rollNumber_04_task1.py   # Main script that connects all modules: training, testing, and plotting
├── train.py                 # Training script with early stopping, perplexity, BLEU score, and checkpointing
├── test.py                  # Interactive answer generation using top-k/top-p decoding
├── model.py                 # Model definition: MiniTransformer (GPT-style decoder)
├── data_utils.py            # Dataset class and preprocessing functions
├── weights/                 # Directory to store model checkpoints
├── README.md                # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib nltk transformers
```

### 2. Prepare the Dataset

Ensure your dataset is split into train and validation DataFrames, with the following columns:

- question_text — the question title or description
- answer_body — the corresponding natural language answer

Customize the tokenization and masking logic in data_utils.py accordingly.

### 3. Train the Model

```bash
python train.py
```

This will:
- Train the MiniTransformer model
- Track training/validation loss and perplexity
- Compute BLEU score after each epoch
- Save the best model in weights/best_model.pt

### 4. Run Evaluation (Interactive)

```bash
python test.py
```

This script:
- Loads the best model from weights/
- Accepts a user question as input
- Generates a natural language answer using top-k/top-p decoding

---

## 📦 Module Descriptions

- model.py  
  Contains the MiniTransformer class — a decoder-only transformer model with positional encodings, causal attention, and output projection layer.

- data_utils.py  
  Defines the QADataset class and tokenization functions. Includes masking for padded tokens and converts text to PyTorch tensors.

- train.py  
  Implements the training loop with early stopping and learning rate scheduling. Tracks:
  - Loss and Perplexity
  - BLEU score on validation set
  - Saves training history for plotting

- test.py  
  Loads the best model checkpoint and runs interactive QA generation with sampling strategies.

- rollNumber_04_task1.py  
  Main script to run the end-to-end pipeline in a notebook or script. Can call training, evaluation, and plotting from here.

---

## 📊 Visualizations

- Training and validation loss and perplexity are saved during training.
- Use matplotlib to visualize after training:

```python
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.plot(history['bleu'], label='BLEU Score')
plt.legend()
plt.show()
```

---

## 🧠 Model Features

- Decoder-only Transformer architecture
- Positional Encoding
- CrossEntropy Loss with padding ignored
- Learning rate scheduling (Cosine Annealing)
- Early stopping to prevent overfitting
- Top-k and top-p sampling for answer generation
- BLEU score evaluation

---

## ✅ Requirements

- Python 3.7+
- PyTorch
- HuggingFace Transformers
- NLTK (for BLEU score)
- Matplotlib (for plotting)

---

## 📌 Notes

- You may adjust max generation length and sampling parameters in test.py
- Make sure vocab size matches between training and loading (keep tokenizer fixed)
- Avoid loading mismatched model/tokenizer versions

---

## 👤 Author

- Roll Number: [Your Roll Number]
- Task: 1 — Generative QA using Transformers
