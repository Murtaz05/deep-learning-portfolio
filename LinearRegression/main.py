from src.data_processing import DataLoader
from src.train import linear_regression_network, train
from src.evaluate import test_model
import os   
import pandas as pd
from src.utils import load_model
import matplotlib.pyplot as plt


os.chdir("/home/murtaza/University_Data/deep_learning/murtaza_msds24040_01/task1")
df_raw = pd.read_csv("./dataset/california_housing_train.csv")
test_df = pd.read_csv("./dataset/california_housing_test.csv")


batch_size = 32  # Define a batch size
hidden_size = 8  
n_epochs = 150
learning_rate = 0.01
dataloader = DataLoader(df_raw, batch_size)
net = linear_regression_network(hidden_size)
trained_model, loss_epoch_tr, loss_epoch_val = train(net, dataloader, n_epochs, learning_rate)

test_model(test_df)

plt.plot(range(n_epochs), loss_epoch_tr, label='Training Loss')
plt.plot(range(n_epochs), loss_epoch_val, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss, lr = {learning_rate}, hidden_size = {hidden_size}')
plt.legend()
plt.show()

