import torch
import pandas as pd
import pickle
from src.utils import l2_loss, compute_gradient, optimization, dump_model
import torch
import torch.nn as nn

# Define the network architecture
class LinearRegressionNetwork(nn.Module):
    def __init__(self, hidden_neurons):
        super(LinearRegressionNetwork, self).__init__()
        self.input_size = 8
        self.hidden_size = hidden_neurons
        self.output_size = 1

        # Define layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False)  # No bias
        self.fc2 = nn.Linear(self.hidden_size, self.output_size, bias=False) 

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def feed_forward(self, x):
            # Implement y = X * w for each layer
            self.z1 = torch.matmul(x, self.fc1.weight.T)  # X * W for first layer
            self.a1 = torch.relu(self.z1)                       # Apply ReLU activation
            x = torch.matmul(self.a1, self.fc2.weight.T)  # X * W for second layer (output)
            return x
    
    def get_params(self):
        # Return the model parameters as a tensor (theta)
        return self.fc1.weight, self.fc2.weight
    
    def update_params(self, new_params):
        # Ensure weight update is done correctly
        with torch.no_grad():
            self.fc1.weight.copy_(new_params[0])
            self.fc2.weight.copy_(new_params[1])

    def backward(self, x, grad_loss):
       
        # Compute gradient for fc2 (output layer)
        dW2 = torch.matmul(grad_loss.T, self.a1)  # dL/dW2
       
        # Backpropagate through ReLU (only pass gradient where z1 > 0)
        dA1 = torch.matmul(grad_loss, self.fc2.weight)  # dL/dA1
        dZ1 = dA1 * (self.z1 > 0).float()  # dA1 * ReLU derivative

        # Compute gradient for fc1 (hidden layer)
        dW1 = torch.matmul(dZ1.T, x)  # dL/dW1

        return dW1, dW2

# Function to create network
def linear_regression_network(hidden_neurons):
    net = LinearRegressionNetwork(hidden_neurons)
    print(f"Network with {hidden_neurons} hidden neurons initialized.")

    # Correctly access model parameters
    theta_shapes = [param.shape for param in net.get_params()]
    print(f"Total parameters (theta): {theta_shapes}")  

    return net


def train(net, dataloader, n_epochs, lr):
    loss_epoch_tr = []  # List to store training loss for each epoch
    loss_epoch_val = []  # List to store validation loss for each epoch

    for epoch in range(n_epochs):
        epoch_train_loss = 0  # To accumulate the training loss for the current epoch

        # Training phase
        for batch_X, batch_Y in dataloader:
            # 1. Feed forward through the network
            y_hat = net.feed_forward(batch_X)

            # 2. Compute loss
            loss = l2_loss(batch_Y, y_hat)
            print(f"Batch size: {batch_X.shape[0]}")

            print(f"  Batch Loss: {loss.item():.6f}") 

            epoch_train_loss += loss.item()  # Accumulate the loss

            # 3. Compute gradients
            dW1, dW2 = compute_gradient(batch_X, batch_Y, y_hat, net)
             # 4. Get current weights
            W1, W2 = net.get_params()

            # 5. Update weights
            W1_new = optimization(lr, dW1, W1)  # Apply optimization step
            W2_new = optimization(lr, dW2, W2)
            net.update_params((W1_new, W2_new)) 

        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / len(dataloader)
        loss_epoch_tr.append(avg_train_loss)

        # Validation phase
        with torch.no_grad():  # Disable gradient computation during validation
            val_y_hat = net.feed_forward(dataloader.X_val)
            val_loss = l2_loss(dataloader.Y_val, val_y_hat)
        loss_epoch_val.append(val_loss.item())

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the trained model
    dump_model(net)

    return net, loss_epoch_tr, loss_epoch_val
