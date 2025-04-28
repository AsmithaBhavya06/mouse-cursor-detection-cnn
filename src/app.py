import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.dataset import DatasetLoader
from src.data.preprocess import preprocess_images
from src.models.cnn_model import CNNModel
from src.training.train import train_model
from src.config import Config

def main():
    # Load and preprocess the dataset
    dataset_loader = DatasetLoader(Config.DATASET_PATH)
    images, labels = dataset_loader.load_data()
    processed_images = preprocess_images(images)

    # Initialize the CNN model
    model = CNNModel()
    model.build_model()

    # Train the model
    train_model(model, processed_images, labels)

if __name__ == "__main__":
    main()