from imports import *
from data_utils import image_preprocessing
from model import create_captioning_model
from train import max_caption_length, vocab_size
from test import generate_caption
# Test_eval

eval_image_path = "/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/Images/667626_18933d713e.jpg"
eval_image_tensor = image_preprocessing(eval_image_path)


captioning_model = create_captioning_model(vocab_size, max_caption_length)
# Load the trained weights
captioning_model.load_weights('/home/murtaza/University_Data/deep_learning/assignment3/murtaza_msds24040_03/Task1/best_model.weights.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
# Generate caption
caption = generate_caption(captioning_model, eval_image_tensor, tokenizer, max_length=max_caption_length)
print(f"Generated Caption: {caption}")
# Visualize the image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(eval_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(caption)
plt.show()
