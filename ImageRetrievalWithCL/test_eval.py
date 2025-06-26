import torch
from model import SiameseNetwork
from torch.utils.data import Dataset, DataLoader
from utils import predict_label
from PIL import Image
from data_utils import transform


import os
print("Current Working Directory:", os.getcwd())
os.chdir('/home/murtaza/University_Data/deep_learning/assignment2/murtaza_msds24040_02')
print(os.getcwd())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SiameseNetwork()  # Create model instance
model.load_state_dict(torch.load("weights.pth"))  # Load saved weights
model.to(device)  # Move to GPU if available
model.eval()  # Set to evaluation mode


img_path1 = '../caltech-101/BACKGROUND_Google/image_0221.jpg'
img_path2 = '../caltech-101/BACKGROUND_Google/image_0004.jpg'
img_path2 = '../caltech-101/anchor/image_0013.jpg'

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


