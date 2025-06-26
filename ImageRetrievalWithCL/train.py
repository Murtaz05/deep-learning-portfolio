import torch
from model import SiameseNetwork
from torch.utils.data import Dataset, DataLoader
from data_utils import load_transformed_batched_data
import torch.optim as optim
from utils import contrastive_loss, metrics, predict_label
import os
# print("Current Working Directory:", os.getcwd())
os.chdir('/home/murtaza/University_Data/deep_learning/assignment2/murtaza_msds24040_02')
# print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 15
batch_size = 16

train_loader, val_loader = load_transformed_batched_data(batch_size=batch_size)


# for plots
train_losses = []
val_losses = []
train_metrics = []  
val_metrics = []

previous_val_loss = float('inf')  # for callback to reduce lr on plateau
best_val_loss = float('inf')  # Track best loss

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_distances, train_labels = [], []

    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device),label.to(device)
        
        optimizer.zero_grad() #to reset the gradients at each batch
        distance = model(img1, img2)
        loss = contrastive_loss(distance, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_distances.append(distance)
        train_labels.append(label)

    train_loss/= len(train_loader)

    train_distances = torch.cat(train_distances)
    train_labels = torch.cat(train_labels)

    train_pred_labels = predict_label(train_distances)
    train_metrics_result = metrics(train_labels, train_pred_labels)

    train_losses.append(train_loss)
    train_metrics.append(train_metrics_result)


    model.eval()
    val_loss = 0
    val_distances, val_labels = [], []

    with torch.no_grad():  # Disable gradient calculations for validation
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            distance = model(img1, img2)
            loss = contrastive_loss(distance, label)
            val_loss += loss.item()
            val_distances.append(distance)
            val_labels.append(label)

    val_loss /= len(val_loader)
       # Convert to single tensor
    val_distances = torch.cat(val_distances)
    val_labels = torch.cat(val_labels)

    # Convert distances to predicted labels
    val_pred_labels = predict_label(val_distances)

    # Compute Validation Metrics
    val_metrics_result = metrics(val_labels, val_pred_labels)

    val_losses.append(val_loss)
    val_metrics.append(val_metrics_result)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Training Metrics: {train_metrics_result}")
    print(f"Validation Metrics: {val_metrics_result}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), "weights.pth")
        print("Model saved!")

    if val_loss >= previous_val_loss:  # If validation loss does NOT improve
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5  # Reduce LR by half
        print(f"Reducing LR to {optimizer.param_groups[0]['lr']}")
    previous_val_loss = val_loss
