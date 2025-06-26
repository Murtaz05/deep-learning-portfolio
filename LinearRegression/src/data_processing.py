import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import os


def split_df(df_raw):
    rows = df_raw.shape[0]
    data_size = int(0.85 * rows)
    df_train = df_raw.iloc[:data_size]
    df_val = df_raw.iloc[data_size:]

    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_val = df_val.drop('target', axis=1)
    Y_val = df_val['target']

    return X_train, Y_train, X_val, Y_val


def set_mean(x):
    return x.mean(axis = 0)

def set_standardDeviation(x):
    return x.std()

def normalize(x, mean, std):
    return (x-mean)/std

class DataLoader:
    def __init__(self,df,batch_size):
        self.data = df
        self.batch_size = batch_size
        self.X_train, self.Y_train, self.X_val, self.Y_val = split_df(self.data)

        self.mean = set_mean(self.X_train)
        self.std = set_standardDeviation(self.X_train)

        # Normalize data
        self.X_train = normalize(self.X_train, self.mean, self.std)
        self.X_val = normalize(self.X_val, self.mean, self.std)

    # Convert to NumPy arrays first
        self.X_train = self.X_train.to_numpy()
        self.Y_train = self.Y_train.to_numpy()
        self.X_val = self.X_val.to_numpy()
        self.Y_val = self.Y_val.to_numpy()

        # Convert to tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32).view(-1, 1)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.Y_val = torch.tensor(self.Y_val, dtype=torch.float32).view(-1, 1)

        # Save normalization parameters
        torch.save({'mean': self.mean, 'std': self.std}, './model/normalization.pkl')

    def __len__(self):
        return int(np.ceil(len(self.X_train) / self.batch_size))

    def __iter__(self):
        self.idx = 0  # Reset batch index for each new iteration
        return self

    
    def __next__(self):
        if self.idx >= len(self.X_train):  
            raise StopIteration  # Stop iteration when all batches are processed

        start_id = self.idx
        end_id = min(self.idx + self.batch_size, len(self.X_train))  # Ensure no out-of-bounds error

        batch_X = self.X_train[start_id:end_id]
        batch_Y = self.Y_train[start_id:end_id]

        self.idx += self.batch_size  # Move to next batch

        if batch_X.size(0) == 0:  
            raise StopIteration  # Skip empty batch

        return batch_X, batch_Y
