from src.data.dataset import DatasetLoader
from src.data.preprocess import preprocess_images
from src.models.cnn_model import CNNModel
from src.utils.helpers import save_model_checkpoint, calculate_accuracy
import tensorflow as tf

def train_model():
    # Load the dataset
    dataset_loader = DatasetLoader()
    train_images, train_labels = dataset_loader.load_training_data()
    val_images, val_labels = dataset_loader.load_validation_data()

    # Preprocess the images
    train_images = preprocess_images(train_images)
    val_images = preprocess_images(val_images)

    # Initialize the CNN model
    model = CNNModel()
    model.build_model()
    model.compile_model()

    # Set up training parameters
    epochs = 50
    batch_size = 32
    checkpoint_path = "model_checkpoints/cnn_model.h5"

    # Create a callback for saving the model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Train the model
    history = model.model.fit(
        train_images, 
        train_labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(val_images, val_labels),
        callbacks=[model_checkpoint_callback]
    )

    # Calculate and print the accuracy on the validation set
    val_accuracy = calculate_accuracy(model.model, val_images, val_labels)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

if __name__ == "__main__":
    train_model()