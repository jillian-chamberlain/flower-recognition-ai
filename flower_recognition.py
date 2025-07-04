print("Starting flower recognition script...")


import subprocess
import sys

# Automatically install required packages
required = ['tensorflow', 'numpy', 'matplotlib']
for package in required:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os
import zipfile
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import urllib.request

# === CONFIGURATION ===
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = './data/train'
FLOWER_INFO_PATH = './flower_info.json'
DATASET_ZIP_URL = 'https://github.com/jillian-chamberlain/flower-recognition-ai/releases/download/v1.0/train.zip'
DATASET_ZIP_PATH = './train.zip'

# === FUNCTION TO DOWNLOAD & EXTRACT DATASET ===
def download_and_extract_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset not found locally. Downloading from {DATASET_ZIP_URL} ...")
        urllib.request.urlretrieve(DATASET_ZIP_URL, DATASET_ZIP_PATH)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall('./data')
        os.remove(DATASET_ZIP_PATH)
        print("Extraction done.")
    else:
        print("Dataset found locally.")

# === CREATE FLOWER INFO FILE IF MISSING ===
def create_flower_info():
    if not os.path.exists(FLOWER_INFO_PATH):
        print("Creating default flower_info.json ...")
        flower_info = {
            "daisy": "Daisies can have white petals, usually with yellow centers.",
            "rose": "Roses are fragrant flowers that come in various different colors.",
            "sunflower": "Sunflowers are large with yellow petals and brown centers.",
            "tulip": "Tulips are colorful, cup-shaped flowers that bloom in spring.",
            "dandelion": "Dandelions have bright yellow flowers and puffball seeds."
        }
        with open(FLOWER_INFO_PATH, 'w') as f:
            json.dump(flower_info, f, indent=4)
    else:
        print("flower_info.json found.")

# === LOAD DATASET FOR TRAINING ===
def load_data():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb',
        shuffle=True)
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb',
        shuffle=True)
    return train_gen, val_gen

# === BUILD CNN MODEL ===
def create_cnn_model(input_shape, num_classes, learning_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === TRAIN AND SELECT BEST MODEL ===
def train_best_model(train_gen, val_gen):
    learning_rates = [0.001, 0.0001]
    best_model = None
    best_val_acc = 0
    for lr in learning_rates:
        print(f"\nTraining with learning rate {lr}")
        model = create_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 3), train_gen.num_classes, lr)
        history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, verbose=1)
        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_model

# === LOAD FLOWER DESCRIPTIONS ===
def load_flower_info():
    with open(FLOWER_INFO_PATH, 'r') as f:
        return json.load(f)

# === PREDICT USER IMAGE ===
def predict_image(model, image_path, class_labels, flower_info):
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    confidence = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    print(f"\nPrediction: {predicted_class} (Confidence: {confidence:.2f})")
    if confidence < 0.5:
        print("Warning: Low confidence. Please try another image.")
    if predicted_class in flower_info:
        print(f"Info: {flower_info[predicted_class]}")
    else:
        print("No additional information available.")

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{predicted_class} ({confidence:.2f})")
    plt.show()

# === MAIN FUNCTION ===
def main():
    print("=== Flower Recognition AI ===\n")
    download_and_extract_dataset()
    create_flower_info()
    train_gen, val_gen = load_data()
    model = train_best_model(train_gen, val_gen)
    flower_info = load_flower_info()
    class_labels = list(train_gen.class_indices.keys())

    while True:
        user_input = input("\nEnter path to flower image to classify (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        predict_image(model, user_input, class_labels, flower_info)

if __name__ == "__main__":
    main()

