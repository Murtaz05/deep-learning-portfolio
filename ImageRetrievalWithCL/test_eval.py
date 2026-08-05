import torch
from model import SiameseNetwork
from torch.utils.data import Dataset, DataLoader
from utils import predict_label
from PIL import Image
from data_utils import transform


import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset", "caltech-101")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SiameseNetwork()  # Create model instance
model.load_state_dict(torch.load("weights.pth"))  # Load saved weights
model.to(device)  # Move to GPU if available
model.eval()  # Set to evaluation mode


img_path1 = os.path.join(DATASET_DIR, "BACKGROUND_Google", "image_0221.jpg")
img_path2 = os.path.join(DATASET_DIR, "BACKGROUND_Google", "image_0004.jpg")
img_path2 = os.path.join(DATASET_DIR, "anchor", "image_0013.jpg")

img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")

img1 = transform(img1).unsqueeze(0).to(device)  # Add batch dimension and move to device
img2 = transform(img2).unsqueeze(0).to(device)  # Add batch dimension and move to device

with torch.no_grad():
    dist = model(img1,img2)
    lab = predict_label(dist)
    if lab.item():
        print("Same label Images")
    else:
        print("Different label Images")


