from imports import *
from model import create_captioning_model
from train import vocab_size, max_caption_length
from data_utils import FlickrDataset, test_data

captioning_model = create_captioning_model(vocab_size, max_caption_length)
# Load the trained weights
captioning_model.load_weights('best_model.weights.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

image_dir = '/home/murtaza/University_Data/deep_learning/assignment3/Task_01_dataset_flicker/Images'

test_dataset = FlickrDataset(image_dir, test_data, tokenizer, max_length=max_caption_length)


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
        predicted_id = np.random.choice(len(probabilities), p=probabilities)

        # predicted_id = tf.argmax(predictions[0, len(input_caption)-1]).numpy()

        word = tokenizer.index_word.get(predicted_id, '<unk>')
        
        next_token = tf.argmax(predictions[0, len(input_caption)-1]).numpy()
        # print("Next token index:", next_token, "Word:", tokenizer.index_word.get(next_token, "<UNK>"))

        if word == 'endsentence':
            break
        input_caption.append(predicted_id)
    
    caption = [tokenizer.index_word.get(i, '') for i in input_caption[1:]]
    return ' '.join(caption)

if __name__ == "__main__":
    for i in range(5):
        image_tensor, _ = test_dataset[i]  # caption is unused
        # image_tensor = tf.keras.applications.resnet50.preprocess_input(image_tensor)

        caption = generate_caption(captioning_model, image_tensor, tokenizer, max_length=max_caption_length)
        print(f"Generated Caption {i+1}: {caption}")
