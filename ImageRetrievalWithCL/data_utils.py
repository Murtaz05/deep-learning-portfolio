from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

import os
# print("Current Working Directory:", os.getcwd())
os.chdir('/home/murtaza/University_Data/deep_learning/assignment2/murtaza_msds24040_02')
# print(os.getcwd())

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

# print(dataset.head())
# print(dataset.shape)



train_set, temp = train_test_split(dataset, test_size=0.2, stratify=dataset["label"], random_state=42)
test_set, val_set = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

train_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)   
val_set.reset_index(drop=True, inplace=True)

# print(train_set.shape, test_set.shape, val_set.shape)



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

def load_transformed_batched_data(batch_size):
    train_loader = DataLoader(train_transformed, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_transformed, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_test_transformed_batched_data(batch_size):
    test_loader = DataLoader(test_transformed, batch_size=batch_size, shuffle=False)
    return test_loader