from imports import *



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
