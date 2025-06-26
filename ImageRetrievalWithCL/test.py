import torch
from model import SiameseNetwork
from torch.utils.data import Dataset, DataLoader
from utils import predict_label
from PIL import Image
from data_utils import load_test_transformed_batched_data
from utils import contrastive_loss, metrics

import os
# print("Current Working Directory:", os.getcwd())
os.chdir('/home/murtaza/University_Data/deep_learning/assignment2/murtaza_msds24040_02')
# print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SiameseNetwork()  # Create model instance
model.load_state_dict(torch.load("weights.pth"))  # Load saved weights
model.to(device)  # Move to GPU if available
model.eval()  # Set to evaluation mode


test_loader = load_test_transformed_batched_data(batch_size=16)

test_loss = 0
test_distances, test_labels = [], []

with torch.no_grad():  # Disable gradient calculations for validation
    for img1, img2, label in test_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        distance = model(img1, img2)
        loss = contrastive_loss(distance, label)
        test_loss += loss.item()
        test_distances.append(distance)
        test_labels.append(label)

test_loss /= len(test_loader)

# Convert to single tensor
test_distances = torch.cat(test_distances)
test_labels = torch.cat(test_labels)

# Convert distances to predicted labels
test_pred_labels = predict_label(test_distances)

# Compute Test Metrics
test_metrics_result = metrics(test_labels, test_pred_labels)

# Print Results
print("\nTest Evaluation Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics_result}")