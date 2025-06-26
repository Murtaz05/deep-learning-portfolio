# %%
# To run on CPU

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %%
import pandas as pd
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np


import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pickle
from tqdm import tqdm  # For showing progress bar (optional)
import math

# %%

caption_file = '/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/captions.txt' 
df = pd.read_csv(caption_file)

# Group and convert to a dictionary directly using pandas
image_to_captions = df.groupby('image')['caption'].apply(list).to_dict()
random.seed(42)

# Get all unique image filenames
all_images = list(image_to_captions.keys())
random.shuffle(all_images)

# Define split sizes
num_train = 1100
num_val = 250
num_test = 250

# Split the image filenames
train_images = all_images[:num_train]
val_images = all_images[num_train:num_train + num_val]
test_images = all_images[num_train + num_val:num_train + num_val + num_test]

# Create separate dicts for each split
train_data = {img: image_to_captions[img] for img in train_images}
val_data = {img: image_to_captions[img] for img in val_images}
test_data = {img: image_to_captions[img] for img in test_images}

# %%
list(test_data.items())[0]


# %%
class FlickrDataset(Dataset):
    def __init__(self, image_dir ,data_dict, tokenizer, max_length):
        self.image_dir = image_dir
        self.data = list(data_dict.items())  # [(img_name, [captions])]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, captions = self.data[idx]
        caption = random.choice(captions)  # randomly pick one caption
        caption = text_preprocessing(caption)

        caption_seq = self.tokenizer.texts_to_sequences([caption])[0]
        caption_seq = pad_sequences([caption_seq], maxlen=self.max_length, padding='post')[0]
        caption_tensor = tf.convert_to_tensor(caption_seq, dtype=tf.int32)

        image_path = os.path.join(self.image_dir, img_name)

        image_tensor = image_preprocessing(image_path)

        return image_tensor, caption_tensor

def image_preprocessing(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Scale to [0, 1]

    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return tf.convert_to_tensor(img, dtype=tf.float32)

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces - as asked a mandatory extra preporcessing step
    text = f"startsentence {text} endsentence"
    return text


# %%
print(text_preprocessing( "A caption: with     punctuation! & symbols."))

# %%
train_data.values()

# %%
# Flatten all captions from train_data
train_captions = []
for caption_list in train_data.values():
    for caption in caption_list:
        train_captions.append(text_preprocessing(caption))


# %%
#Vocabulary creation
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_captions)

vocab_size = len(tokenizer.word_index) + 1  # +1 for padding (index 0)
print("Vocabulary Size:", vocab_size)


# Tokenize the captions
train_sequences = tokenizer.texts_to_sequences(train_captions)

# Find the maximum caption length
max_caption_length = max(len(seq) for seq in train_sequences)
print("Max caption length:", max_caption_length)


# %%
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# %%
#checking the tokenizer
print(train_captions[1])
seq = tokenizer.texts_to_sequences([train_captions[1]])[0]
print(seq)
print(tokenizer.sequences_to_texts([seq]))


# %%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

# --- Image Feature Extractor (ResNet50) ---
def create_image_embedding_model():
    # Input for image
    image_input = Input(shape=(224, 224, 3))  # ResNet50 expects (224, 224, 3)
    
    # Use the ResNet50 pre-trained model (without the top classification layer)
    base_model = ResNet50(weights='imagenet', include_top=False)(image_input)
    
    # Apply GlobalAveragePooling to the output of the last convolutional layer
    x = GlobalAveragePooling2D()(base_model)
    
    # Add a Dense layer to project the features to a fixed-size embedding space
    x = Dense(256, activation='relu')(x)
    
    # Add a dropout layer for regularization
    x = Dropout(0.5)(x)  # Adjust dropout rate as needed
    
    # Create a final image feature model
    image_embedding_model = Model(inputs=image_input, outputs=x)
    
    return image_embedding_model


# %%
image_embedding_model = create_image_embedding_model()
image_embedding_model.summary()

# %%
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model

# --- Text Embedding Model ---
def create_text_embedding_model(vocab_size, max_caption_length, embedding_dim=256):
    # Input for captions (sequence of word indices)
    caption_input = Input(shape=(max_caption_length,))
    
    # Embedding layer to convert word indices into dense vectors
    caption_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(caption_input)
    
    # LSTM layer to process the sequence of embeddings
    caption_lstm = LSTM(256, return_sequences=False)(caption_embedding)
    
    # Dense layer to output a fixed-size vector (optional)
    caption_output = Dense(256, activation='relu')(caption_lstm)
    
    # Create the text embedding model
    text_embedding_model = Model(inputs=caption_input, outputs=caption_output)
    
    return text_embedding_model


# %%
text_embedding_model = create_text_embedding_model(vocab_size, max_caption_length)
text_embedding_model.summary()

# %%
from tensorflow.keras.layers import Concatenate, RepeatVector, LSTM, TimeDistributed, Dense, Input
from tensorflow.keras.models import Model

def create_captioning_model(vocab_size, max_caption_length, embedding_dim=256):
    # Image and Text Embedding Models
    image_embedding_model = create_image_embedding_model()
    text_embedding_model = create_text_embedding_model(vocab_size, max_caption_length, embedding_dim)

    # Inputs
    image_input = image_embedding_model.input
    caption_input = text_embedding_model.input

    # Outputs (both are (None, 256))
    image_features = image_embedding_model.output
    text_features = text_embedding_model.output

    # Merge embeddings
    merged_features = Concatenate(axis=-1)([image_features, text_features])  # (None, 512)

    # Repeat vector to match caption length
    repeated = RepeatVector(max_caption_length)(merged_features)  # (None, max_caption_length, 512)

    # LSTM Decoder
    x = LSTM(512, return_sequences=True)(repeated)
    x = LSTM(512, return_sequences=True)(x)
    x = TimeDistributed(Dense(512, activation='relu'))(x)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    # Final Model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model



# %%
captioning_model = create_captioning_model(vocab_size, max_caption_length)
captioning_model.summary()

# %%

# --- Data Preparation ---
image_dir = '/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/Images'
train_dataset = FlickrDataset(image_dir, train_data, tokenizer, max_length=max_caption_length)
val_dataset = FlickrDataset(image_dir, val_data, tokenizer,  max_length=max_caption_length)
test_dataset = FlickrDataset(image_dir, test_data, tokenizer, max_length=max_caption_length)


# %%
def dataset_generator(dataset):
    for i in range(len(dataset)):
        image, caption = dataset[i]
        input_caption = caption[:-1]
        target_caption = caption[1:]
            # Pad input_caption to length 35
        input_caption = tf.pad(input_caption, [[0, 1]], constant_values=0)  # Padding to length 35

        # Pad target_caption to length 35
        target_caption = tf.pad(target_caption, [[0, 1]], constant_values=0)  # Padding to length 35

        yield ((image, input_caption), target_caption)
output_signature = (
    (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),           # image
        tf.TensorSpec(shape=(max_caption_length,), dtype=tf.int32) # input caption
    ),
    tf.TensorSpec(shape=(max_caption_length,), dtype=tf.int32)     # target caption
)



# %%

BATCH_SIZE = 32
# Create TensorFlow datasets from your FlickrDataset instance
train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train_dataset),  # assuming train_dataset is an instance of FlickrDataset
    output_signature=output_signature
).shuffle(1000).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

val_tf_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(val_dataset),  # assuming val_dataset is an instance of FlickrDataset
    output_signature=output_signature
).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)



# %%
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


class BLEUCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data, tokenizer, max_len):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.train_bleus = []
        self.val_bleus = []

    def set_model(self, model):
        self.model_ref = model  # This is how Keras passes the model reference

    def on_epoch_end(self, epoch, logs=None):
        train_bleu = self._compute_bleu(self.train_data)
        val_bleu = self._compute_bleu(self.val_data)

        self.train_bleus.append(train_bleu)
        self.val_bleus.append(val_bleu)

        print(f"\nEpoch {epoch+1} - Train BLEU: {train_bleu:.4f} | Val BLEU: {val_bleu:.4f}")

    def _compute_bleu(self, dataset):

        bleu_scores = []
        smooth_fn = SmoothingFunction().method1

        for (img_batch, input_caption), target_caption in dataset.take(5):  # Use a few batches to save time
            preds = self.model_ref.predict_on_batch([img_batch, input_caption])
            predicted_ids = tf.argmax(preds, axis=-1).numpy()

            for pred, target in zip(predicted_ids, target_caption.numpy()):
                pred_caption = self._decode(pred)
                target_caption_text = self._decode(target)

                ref = [target_caption_text.split()]
                hyp = pred_caption.split()
                bleu = sentence_bleu(ref, hyp, smoothing_function=smooth_fn, weights=(0.5, 0.5))
                bleu_scores.append(bleu)

        return np.mean(bleu_scores)

    def _decode(self, seq):
        words = []
        for idx in seq:
            if idx == 0:
                continue
            word = self.tokenizer.index_word.get(idx, "")
            if word == '<end>':
                break
            words.append(word)
        return ' '.join(words)

    def on_train_end(self, logs=None):
        import matplotlib.pyplot as plt
        plt.plot(self.train_bleus, label="Train BLEU")
        plt.plot(self.val_bleus, label="Val BLEU")
        plt.xlabel("Epochs")
        plt.ylabel("BLEU Score")
        plt.title("BLEU Score over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


# %%
#callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

# to decay and reduce LR
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,             # Reduce LR by 50%
    patience=2,             # Wait 2 epochs with no improvement
    min_lr=1e-6,            # Don't go below this
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='best_model.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,            # Stop if no improvement after 5 epochs
    restore_best_weights=True,
    verbose=1
)



# %%
epochs = 10
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
captioning_model.compile(optimizer=optimizer, loss=loss_fn)

train_steps = math.ceil(len(train_dataset) / BATCH_SIZE)
val_steps = math.ceil(len(val_dataset) / BATCH_SIZE)

bleu_callback = BLEUCallback(train_tf_dataset, val_tf_dataset, tokenizer, max_caption_length)

# Train the model
history = captioning_model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[bleu_callback, lr_reducer, checkpoint, early_stopping] 

)


# %%

# Plot training & validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.show()

# %%
# Load the trained weights
captioning_model.load_weights('best_model.weights.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# %%
def generate_caption(model, image_tensor, tokenizer, max_length):
    input_caption = [tokenizer.word_index['startsentence']]
    
    for _ in range(max_length):
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [input_caption], maxlen=max_length, padding='post'
        )
        
        predictions = model.predict(
            [tf.expand_dims(image_tensor, 0), input_seq], verbose=0
        )
        
        # Visualize the full prediction distribution
        # print(f"Predictions at step {_}: {predictions[0, len(input_caption)-1]}")
        # Sort predictions by probability to check the top tokens
        # sorted_indices = np.argsort(predictions[0, len(input_caption)-1])[::-1]
        # print("Top 5 tokens (sorted by probability):")
        # for idx in sorted_indices[:5]:
        #     print(f"Token: {tokenizer.index_word.get(idx, '<unk>')}, Probability: {predictions[0, len(input_caption)-1][idx]}")
        # Sampling from the probability distribution (instead of using argmax)
        probabilities = tf.nn.softmax(predictions[0, len(input_caption)-1]).numpy()
        # predicted_id = np.random.choice(len(probabilities), p=probabilities)

        predicted_id = tf.argmax(predictions[0, len(input_caption)-1]).numpy()

        word = tokenizer.index_word.get(predicted_id, '<unk>')
        
        next_token = tf.argmax(predictions[0, len(input_caption)-1]).numpy()
        # print("Next token index:", next_token, "Word:", tokenizer.index_word.get(next_token, "<UNK>"))

        if word == 'endsentence':
            break
        input_caption.append(predicted_id)
    
    caption = [tokenizer.index_word.get(i, '') for i in input_caption[1:]]
    return ' '.join(caption)


# %%
# Example: show 5 generated captions from test set
for i in range(5):
    image_tensor, _ = test_dataset[i]  # caption is unused
    # image_tensor = tf.keras.applications.resnet50.preprocess_input(image_tensor)

    caption = generate_caption(captioning_model, image_tensor, tokenizer, max_length=max_caption_length)
    print(f"Generated Caption {i+1}: {caption}")


# %%
# Test_eval

eval_image_path = "/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/Images/667626_18933d713e.jpg"
eval_image_tensor = image_preprocessing(eval_image_path)

# Load the trained weights
captioning_model.load_weights('best_model.weights.h5')

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


# %%



