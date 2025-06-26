from imports import *

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
        plt.plot(self.train_bleus, label="Train BLEU")
        plt.plot(self.val_bleus, label="Val BLEU")
        plt.xlabel("Epochs")
        plt.ylabel("BLEU Score")
        plt.title("BLEU Score over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()



