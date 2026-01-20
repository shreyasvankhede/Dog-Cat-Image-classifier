import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def clean_dataset(path):
    removed = 0
    total = 0
    
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            total += 1
            
            try:
                img = tf.io.read_file(file_path)
                tf.image.decode_image(img, channels=3)
            except:
                os.remove(file_path)
                removed += 1
    
    print(f"Removed {removed} bad images out of {total}")


def load_dataset(path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    train_ds = train_ds.map(lambda x, y: (mobilenet_preprocess(x), y))
    val_ds = val_ds.map(lambda x, y: (mobilenet_preprocess(x), y))

    return train_ds, val_ds


def train_model():
    clean_dataset("data")

    train_ds, val_ds = load_dataset("data")

    base_model=MobileNetV2(
        input_shape=(128,128,3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable=False #to freeze trained weights

    model=models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1,activation='sigmoid')
    ])

    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"] 
    )

    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
    )

    model.save("cat_dog_classifier_transfer.h5")

def load_model(path):
    if not os.path.isfile(path):
        return -1
    return keras_load_model(path)


def preprocess_input(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = mobilenet_preprocess(img)
    img = np.expand_dims(img, axis=0)

    return img

