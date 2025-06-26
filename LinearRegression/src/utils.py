import torch
import pickle

# L2 loss function
def l2_loss(groundTruth, y_hat):
    # Compute L2 loss (Mean Squared Error)
    loss = torch.mean((groundTruth - y_hat) ** 2)/2
    return loss

def compute_gradient(batch_X, batch_Y, y_hat, net):
    N = batch_Y.size(0)
    grad_loss = (y_hat - batch_Y)/N   # dL/dy_hat
    return net.backward(batch_X, grad_loss)

def optimization(lr, grad, params):
    return params - lr * grad 

def dump_model(model):
    with open('./model/model.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model():
    with open('./model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)



