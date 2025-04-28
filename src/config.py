# Configuration settings for the mouse cursor detection CNN project

import os

# Dataset paths
DATASET_PATH = os.path.join("data", "dataset")
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train_images")
TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "train_labels")
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test_images")
TEST_LABELS_PATH = os.path.join(DATASET_PATH, "test_labels")

# Model parameters
MODEL_SAVE_PATH = os.path.join("models", "cnn_model.h5")
INPUT_SHAPE = (64, 64, 3)  # Example input shape for CNN
NUM_CLASSES = 10  # Adjust based on the number of classes in your dataset

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data augmentation settings
DATA_AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}