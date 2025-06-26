
import torch
from src.utils import l2_loss, load_model
from src.data_processing import normalize
import pandas as pd

def test_model(test_df):

    test_df.dropna(inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Load trained model
    X_test, Y_test = split_X_Y(test_df)

    norm_params = torch.load('./model/normalization.pkl',weights_only=False)
    mean, std = norm_params['mean'], norm_params['std']

    X_test = normalize(X_test, mean, std)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    loaded_model = load_model()
    # Forward pass
    with torch.no_grad():  # No gradients needed for inference
        y_hat = loaded_model.feed_forward(X_test)

    # Compute test loss
    loss = l2_loss(Y_test, y_hat)
    print(f"Test Loss: {loss.item()}")
    return loss


def split_X_Y(test_data):
    X_test = test_data.drop('target', axis=1)
    Y_test = test_data['target']
    return X_test, Y_test

