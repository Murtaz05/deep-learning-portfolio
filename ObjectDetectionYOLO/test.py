from ultralytics import YOLO

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
    pass  # This triggers inference an