# To run on CPU

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



from imports import *
from data_utils import train_data, val_data ,FlickrDataset, dataset_generator, image_preprocessing, text_preprocessing, BLEUCallback
from model import create_captioning_model


# Flatten all captions from train_data
train_captions = []
for caption_list in train_data.values():
    for caption in caption_list:
        train_captions.append(text_preprocessing(caption))
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
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# --- Model Creation ---
captioning_model = create_captioning_model(vocab_size, max_caption_length)


# --- Data Preparation ---
image_dir = '/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/Images'
train_dataset = FlickrDataset(image_dir, train_data, tokenizer, max_length=max_caption_length)
val_dataset = FlickrDataset(image_dir, val_data, tokenizer,  max_length=max_caption_length)


output_signature = (
    (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),           # image
        tf.TensorSpec(shape=(max_caption_length,), dtype=tf.int32) # input caption
    ),
    tf.TensorSpec(shape=(max_caption_length,), dtype=tf.int32)     # target caption
)


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
if __name__ == "__main__":

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


    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.grid(True)
    plt.show()