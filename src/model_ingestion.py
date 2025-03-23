import os
import numpy as np
import pandas as ps
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)
        return False  # Image is not corrupted
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image found: {image_path}")
        return True  # Image is corrupted


def delete_or_keep():
    categories = ['cats', 'dogs']
    for i in categories:
        folder_path = f'cats_vs_dogs/{i}'

        clean_images = []
        total_images = []
        corrupted_images = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            total_images.append(image_path)
            if is_image_corrupted(image_path):
                corrupted_images.append(image_path)

        for image_path in corrupted_images:
            os.remove(image_path)
            print(f"Deleted corrupted image: {image_path}")

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            clean_images.append(image_path)

        print(f'Total images of {i} before Cleansing: ', len(total_images))
        print(f'Total images of {i} after Cleansing: ', len(clean_images))
        print(f"Total corrupted images of {i} found: {len(corrupted_images)}")


delete_or_keep()


def train_test_data():
    datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2, height_shift_range=0.1, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    train_generator = datagen.flow_from_directory('cats_vs_dogs', target_size=(224, 224), batch_size=16,
                                                  class_mode='binary', subset='training')

    valid_generator = datagen.flow_from_directory('cats_vs_dogs', target_size=(224, 224), batch_size=16,
                                                  class_mode='binary', subset='validation')

    return train_generator, valid_generator


if __name__ == '__main__':
    obj = train_test_data()