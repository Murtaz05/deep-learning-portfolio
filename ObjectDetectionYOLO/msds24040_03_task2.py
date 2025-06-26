# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import torch
from ultralytics import YOLO


# %%

# Define paths
image_dir = "/home/murtaza/University_Data/deep_learning/assignment3/Task_02_vehicles_dataset/images"
label_dir = "/home/murtaza/University_Data/deep_learning/assignment3/Task_02_vehicles_dataset/labels"

# Get all image files
image_files = sorted(os.listdir(image_dir))
len(image_files)
label_files = sorted(os.listdir(label_dir))
len(label_files)

# %%
# Optional sanity check
for img_file, lbl_file in zip(image_files, label_files):
    img_name = os.path.splitext(img_file)[0]
    lbl_name = os.path.splitext(lbl_file)[0]
    if img_name != lbl_name:
        print(f"Mismatch: {img_name} vs {lbl_name}")


# %%
data = []
for img_file, lbl_file in zip(image_files, label_files):
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, lbl_file)
    data.append([img_path, lbl_path])
df = pd.DataFrame(data, columns=["image_path", "label_path"])
df.to_csv("TrafficDetectionData.csv", index=False)


# %%

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# %%

# Get all file extensions
file_extensions = set([os.path.splitext(f)[1] for f in os.listdir(image_dir)])
print("Unique file extensions found:", file_extensions)


# %%


# Create YOLO-style folder structure
base_dir = "yolo_dataset"
for split in ["train", "val"]:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

# Function to copy files to YOLO folder
def copy_files(df, split):
    for _, row in df.iterrows():
        img_dest = os.path.join(base_dir, "images", split, os.path.basename(row["image_path"]))
        lbl_dest = os.path.join(base_dir, "labels", split, os.path.basename(row["label_path"]))
        shutil.copy(row["image_path"], img_dest)
        shutil.copy(row["label_path"], lbl_dest)

# Copy training and validation data
copy_files(train_df, "train")
copy_files(val_df, "val")


# %%
data_yaml = """
path: /home/murtaza/University_Data/deep_learning/assignment3/murtaza_msds24040_03/Task2/yolo_dataset
train: images/train
val: images/val

nc: 1
names: ['vehicle']
"""

with open("/home/murtaza/University_Data/deep_learning/assignment3/murtaza_msds24040_03/Task2/yolo_dataset/data.yaml", "w") as f:
    f.write(data_yaml.strip())

print("data.yaml created.")


# %%
torch.cuda.empty_cache()


# %%

# Train YOLOv8-nano
model_nano = YOLO('yolov8n.pt')
model_nano.train(data='yolo_dataset/data.yaml', epochs=50, imgsz=640,name='yolov8n_train')

# Train YOLOv8-small
model_small = YOLO('yolov8s.pt')
model_small.train(data='yolo_dataset/data.yaml', epochs=50, imgsz=640,name='yolov8s_train')


# %% [markdown]
# TESTING

# %%

# Load the best models from training
model_nano = YOLO('runs/detect/yolov8n_train/weights/best.pt')  # Adjust the path to your run
model_small = YOLO('runs/detect/yolov8s_train/weights/best.pt')  # Adjust the path to your run

# You can print a quick summary of both models
print(f"Nano Model: {model_nano}")
print(f"Small Model: {model_small}")

# %%
# Evaluate models
metrics_nano = model_nano.val(name = 'yolov8n_val')
metrics_small = model_small.val(name = 'yolov8s_val')

# Print metrics for both models
print("Metrics for Nano Model:", metrics_nano)
print("Metrics for Small Model:", metrics_small)


# %%
# For nano model
results_nano = model_nano.predict(
    source='yolo_dataset/images/test_images',
    stream=True,
    save=True,
    conf=0.3,
    name='yolov8n_test'
)

for r in results_nano:
    pass  # This triggers inference and saving

# For small model
results_small = model_small.predict(
    source='yolo_dataset/images/test_images',
    stream=True,
    save=True,
    conf=0.3,
    name='yolov8s_test'
)

for r in results_small:
    pass  # This triggers inference and saving

# This will save the results with bounding boxes in separate folders like:
# runs/detect/predict_nano/ and runs/detect/predict_small/


# %%
image_path = '/home/murtaza/University_Data/deep_learning/assignment3/Task_02_vehicles_dataset/images/00 (103).jpg'

label_path = '/home/murtaza/University_Data/deep_learning/assignment3/Task_02_vehicles_dataset/labels/00 (103).txt'

# visualize_prediction_vs_groundtruth(image_path, label_path)

# %%
model_nano.predict(
    source=image_path,
    save=True,
    conf=0.3,
    name='eval'
)

# %%



