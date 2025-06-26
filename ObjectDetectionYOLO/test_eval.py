from ultralytics import YOLO

# Load the best models from training
model_nano = YOLO('runs/detect/yolov8n_train/weights/best.pt')  # Adjust the path to your run
model_small = YOLO('runs/detect/yolov8s_train/weights/best.pt')  # Adjust the path to your run


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
