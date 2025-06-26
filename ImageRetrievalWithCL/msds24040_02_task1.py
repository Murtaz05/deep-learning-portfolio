# %%
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import os
print("Current Working Directory:", os.getcwd())
os.chdir('/home/murtaza/University_Data/deep_learning/assignment2/murtaza_msds24040_02')
print(os.getcwd())

# %% [markdown]
# data_utils

# %%


dataset_path = "../caltech-101"
image_paths = []
labels = []
classes = os.listdir(dataset_path)
for class_label in classes:
    class_path = os.path.join(dataset_path, class_label)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        image_paths.append(image_path)
        labels.append(class_label)


dataset = pd.DataFrame({"image_path": image_paths, "label": labels})

print(dataset.head())
print(dataset.shape)

# %%

train_set, temp = train_test_split(dataset, test_size=0.2, stratify=dataset["label"], random_state=42)
test_set, val_set = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)   
val_set.reset_index(drop=True, inplace=True)

print(train_set.shape, test_set.shape, val_set.shape)




# %%

class SNDataLoader(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1 = self.data["image_path"].iloc[idx]
        img1 = Image.open(img_path1).convert("RGB")
        label1 = self.data["label"].iloc[idx]
            
        if torch.rand(1).item() < 0.5:  # 50% same label
            same_class = self.data[self.data["label"] == label1].drop(idx)  
            idx2 = torch.randint(len(same_class), (1,)).item()
            idx2 = same_class.index[idx2]
        else: 
            different_class = self.data[self.data["label"] != label1]  
            idx2 = torch.randint(len(different_class), (1,)).item()
            idx2 = different_class.index[idx2]
        
        img_path2 = self.data["image_path"].iloc[idx2]

        img2 = Image.open(img_path2).convert("RGB")
        label2 = self.data["label"].iloc[idx2]
        
        label = label1 == label2
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2 ,label
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean =[0.485 , 0.456 , 0.406] , # ImageNet stats
        std =[0.229 , 0.224 , 0.225])])

train_transformed = SNDataLoader(train_set, transform=transform)
test_transformed = SNDataLoader(test_set, transform=transform)
val_transformed = SNDataLoader(val_set, transform=transform)


# %%
train_set

# %%
train_transformed[9][2]

# %%
len(train_transformed), len(train_set), len(val_transformed)

# %%
same = 0
for data in val_transformed:
    if data[2] == 0.0:
        same += 1
print(same)


# %%
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)

    def forward_one(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)
    
    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        distance = torch.norm(emb1 - emb2, p=2, dim=1)  # Euclidean distance

        return distance



# %%
def contrastive_loss(d, y, alpha=1.0):
    loss = torch.mean(y * d.pow(2) +  (1 - y) * torch.clamp(alpha - d, 0).pow(2))
    return loss

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# %% [markdown]
# Training

# %%
def predict_label(d, threshold=0.52): #because mean distance is 0.5142
    return (d < threshold).float()

# %%

def accuracy(true_label, pred_label):
    correct_pred = torch.sum(true_label == pred_label).float()
    return round(correct_pred.item() / len(true_label), 3)

def precision(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).float()
    fp = torch.sum((true_label == 0) & (pred_label == 1)).float()
    return round((tp / (tp + fp)).item(), 3) if (tp + fp) > 0 else 0

def recall(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).float()
    fn = torch.sum((true_label == 1) & (pred_label == 0)).float()
    return round((tp / (tp + fn)).item(), 3) if (tp + fn) > 0 else 0

def f1_score(true_label, pred_label):
    prec = precision(true_label, pred_label)
    rec = recall(true_label, pred_label)
    return round(2 * (prec * rec) / (prec + rec), 3) if (prec + rec) > 0 else 0

def confusion_matrix(true_label, pred_label):
    tp = torch.sum((true_label == 1) & (pred_label == 1)).item()
    tn = torch.sum((true_label == 0) & (pred_label == 0)).item()
    fp = torch.sum((true_label == 0) & (pred_label == 1)).item()
    fn = torch.sum((true_label == 1) & (pred_label == 0)).item()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def classification_rep(true_label, pred_label):
    return classification_report(true_label.cpu().numpy(), pred_label.cpu().numpy())

def metrics(true_label, pred_label):
    acc = accuracy(true_label, pred_label)
    prec = precision(true_label, pred_label)
    rec = recall(true_label, pred_label)
    f1 = f1_score(true_label, pred_label)
    cm = confusion_matrix(true_label, pred_label)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": cm
    }

# %%

model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 15
batch_size = 16

train_loader = DataLoader(train_transformed, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_transformed, batch_size=batch_size, shuffle=False)


# %%
torch.cuda.empty_cache()  # Releases unused memory


# %%
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


# %%

# Extract loss values
epochs = list(range(1, num_epochs + 1))
train_acc = [m["Accuracy"] for m in train_metrics]
val_acc = [m["Accuracy"] for m in val_metrics]

train_prec = [m["Precision"] for m in train_metrics]
val_prec = [m["Precision"] for m in val_metrics]

train_rec = [m["Recall"] for m in train_metrics]
val_rec = [m["Recall"] for m in val_metrics]

train_f1 = [m["F1 Score"] for m in train_metrics]
val_f1 = [m["F1 Score"] for m in val_metrics]


# %%
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss", marker="o")
plt.plot(epochs, val_losses, label="Val Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, val_acc, label="Val Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_prec, label="Train Precision", marker="o")
plt.plot(epochs, val_prec, label="Val Precision", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.title("Precision per Epoch")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_rec, label="Train Recall", marker="o")
plt.plot(epochs, val_rec, label="Val Recall", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.title("Recall per Epoch")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_f1, label="Train F1 Score", marker="o")
plt.plot(epochs, val_f1, label="Val F1 Score", marker="o")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("F1 Score per Epoch")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# validation to find hyper parameter - Threshold

# %%
model = SiameseNetwork()  # Create model instance
model.load_state_dict(torch.load("weights.pth"))  # Load saved weights
model.to(device)  # Move to GPU if available


# %%
all_distances = []
all_labels = []

# Ensure the model is in evaluation mode
model.eval()

with torch.no_grad():
    for img1, img2, label in val_loader:  
        img1, img2 , label= img1.to(device), img2.to(device), label.to(device)
        distance = model(img1, img2)
        
        all_distances.append(distance)
        all_labels.append(label)

# Convert list of tensors into a single tensor
all_distances = torch.cat(all_distances)
all_labels = torch.cat(all_labels)


# %%

print(f"Min Distance: {all_distances.min().item():.4f}")
print(f"Max Distance: {all_distances.max().item():.4f}")
print(f"Mean Distance: {all_distances.mean().item():.4f}")
print(f"Median Distance: {torch.median(all_distances):.4f}")

# %%

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(all_distances.cpu().numpy(), bins=30, kde=True)
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Euclidean Distances")
plt.show()


# %%

def predict_label(d, threshold=0.52): #because mean distance is 0.5142
    return (d < threshold).float()

# %%
pred_labels = predict_label(all_distances)

# %%
metrics(true_label=all_labels, pred_label=pred_labels)

# %%
classification_rep(all_labels,pred_labels)

# %%





model = SiameseNetwork()  # Create model instance
model.load_state_dict(torch.load("weights.pth"))  # Load saved weights
model.to(device)  # Move to GPU if available
model.eval()  # Set to evaluation mode


test_loader = DataLoader(test_transformed, batch_size=batch_size, shuffle=False)

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