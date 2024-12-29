import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import string


class ImageFeatureExtractor:
    def __init__(self):
        self.model = self.create_inception_model()

    # Create the InceptionV3 model without the top layer
    def create_inception_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
        return model

    # Preprocess image for InceptionV3 model
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")  # Convert image to RGB to ensure 3 channels
        img = img.resize((299, 299))  # Resize to 299x299 (expected by InceptionV3)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = tf.keras.applications.inception_v3.preprocess_input(img)  # Preprocess for InceptionV3
        return img

    # Extract features for a given image
    def extract_features(self, image_path):
        try:
            img = self.preprocess_image(image_path)
            features = self.model.predict(img)
            return features.flatten()  # Flatten the features immediately
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    # Extract features for the first `limit` images in the directory
    def extract_features_for_directory(self, image_dir, limit=100):
        image_features = {}
        img_files = os.listdir(image_dir)[:limit]
        for img_name in img_files:
            img_path = os.path.join(image_dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping unsupported file: {img_name}")
                continue

            img_id = os.path.splitext(img_name)[0]  # Remove file extension for unique ID
            features = self.extract_features(img_path)
            if features is not None:
                image_features[img_id] = features  # Store flattened features
        return image_features


class CaptionProcessor:
    def __init__(self, captions_file, image_ids):
        self.captions_file = captions_file
        self.image_ids = image_ids  # Limit to selected image IDs
        self.captions_dict = self.load_and_preprocess_captions()
        self.tokenizer = self.create_tokenizer()
        self.max_len = self.calculate_max_len()

    def load_and_preprocess_captions(self):
        with open(self.captions_file, 'r') as file:
            raw_captions = file.read()

        captions_dict = {}
        table = str.maketrans('', '', string.punctuation)

        for line in raw_captions.split('\n'):
            tokens = line.split(',')

            # Continue if the line is too short to contain a valid caption
            if len(tokens) < 2:
                continue

            image_id = tokens[0].split('.')[0]  # Extract image ID
            caption = ' '.join(tokens[1:]).strip()

            # Remove punctuation and check for empty captions
            caption = caption.lower().translate(table)
            if not caption:  # Skip empty captions
                continue

            # Capitalize the first letter of the caption
            caption = caption[0].upper() + caption[1:]

            if image_id in self.image_ids:  # Only use captions for selected images
                if image_id not in captions_dict:
                    captions_dict[image_id] = []
                captions_dict[image_id].append('startseq ' + caption + ' endseq')

        return captions_dict

    def create_tokenizer(self):
        all_captions = []
        for key in self.captions_dict:
            all_captions.extend(self.captions_dict[key])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        return tokenizer

    def calculate_max_len(self):
        return max(len(caption.split()) for captions in self.captions_dict.values() for caption in captions)

    def create_sequences(self, image_features, vocab_size):
        X1, X2, y = [], [], []

        for key, captions_list in self.captions_dict.items():
            if key not in image_features:  # Skip keys not in image_features
                continue

            for caption in captions_list:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(image_features[key])  # Use flattened image features
                    X2.append(in_seq)
                    y.append(out_seq)

        return np.array(X1), np.array(X2), np.array(y)

def generate_caption(model, image_feature, tokenizer, max_len):
        in_text = 'startseq'
        for _ in range(max_len):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_len)
            yhat = model.predict([image_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat, None)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text
        # return in_text.replace('startseq ', '').replace(' endseq', '')