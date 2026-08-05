# Linear Regression from Scratch (PyTorch)

A neural-network implementation of linear regression on the California Housing dataset, built to understand the training loop from first principles rather than calling `sklearn.LinearRegression`.

## What it covers

- Custom `DataLoader`/dataset split (`src/data_processing.py`)
- A small feed-forward network trained as a linear regressor (`src/train.py`)
- Manual hyperparameter sweep over learning rate, hidden layer size, and epoch count
- Debugging exploding gradients: at `lr=0.9` the batch loss goes to `NaN` after a few iterations

## Results

| Epochs | LR | Hidden size | Train loss | Val loss | Test loss |
|---|---|---|---|---|---|
| 20 | 0.01 | 2 | 0.400 | 0.405 | 248953504.0 |
| 20 | 0.00001 | 2 | 3.335 | 3.296 | 317452.0 |
| 20 | 0.01 | 8 | 0.288 | 0.301 | 286201760.0 |
| 20 | 0.00001 | 8 | 2.538 | 2.485 | 15192609.0 |
| 150 | 0.00001 | 2 | 2.024 | 2.015 | 12722010.0 |
| 150 | 0.00001 | 8 | 0.874 | 0.893 | 6733946.5 |
| **150** | **0.01** | **8** | **0.240** | **0.261** | **211552656.0** |

Best validation loss came from `epochs=150, lr=0.01, hidden_size=8` (bolded) — that's the configuration `main.py` runs by default. Loss curve plots for each sweep are in `plots/`.

**Key finding:** the huge gap between train/val loss (~0.25) and test loss (in the hundreds of millions) reflects the target variable's scale (house prices, unnormalized) — MSE on raw dollar values dominates the metric even when the model is fitting reasonably well on normalized loss.

## Dataset

[California Housing Prices](https://storage.googleapis.com/mledu-datasets/california_housing_train.csv) ([test split](https://storage.googleapis.com/mledu-datasets/california_housing_test.csv)) — not included in this repo. Download both CSVs into a `dataset/` folder here:

```
LinearRegression/
  dataset/
    california_housing_train.csv
    california_housing_test.csv
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Project structure

```
main.py              # entry point: loads data, trains, evaluates, plots loss curves
src/
  data_processing.py  # DataLoader + train/val split
  train.py             # model definition + training loop
  evaluate.py          # test-set evaluation
  utils.py             # model save/load helpers
model/
  model.pkl            # saved trained model
  normalization.pkl    # saved feature normalization stats
plots/                 # loss curves for each hyperparameter combination
```

## Stack

PyTorch, pandas, matplotlib.
