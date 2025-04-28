from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def preprocess_dataset(image_dir, target_size=(64, 64), augment=False):
    images = []
    labels = []
    
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = load_and_preprocess_image(image_path, target_size)
                images.append(image)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        return datagen.flow(images, labels, batch_size=32), len(images)
    
    return images, labels

def normalize_images(images):
    return images.astype('float32') / 255.0  # Normalize to [0, 1]