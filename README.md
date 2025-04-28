# Mouse Cursor Detection using CNN

This project implements a mouse cursor detection system using a Convolutional Neural Network (CNN). The goal is to accurately detect and track the mouse cursor based on input images.

## Project Structure

```
mouse-cursor-detection-cnn
├── src
│   ├── data
│   │   ├── dataset.py        # Handles loading and managing the dataset
│   │   └── preprocess.py     # Contains functions for image preprocessing
│   ├── models
│   │   └── cnn_model.py      # Defines the CNN architecture
│   ├── training
│   │   └── train.py          # Responsible for training the CNN model
│   ├── utils
│   │   └── helpers.py        # Utility functions for various tasks
│   ├── app.py                # Main entry point for the application
│   └── config.py             # Configuration settings for the project
├── requirements.txt           # Lists project dependencies
├── README.md                  # Documentation for the project
└── .gitignore                 # Specifies files to ignore by Git
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd mouse-cursor-detection-cnn
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. Then, install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   Ensure that your dataset is structured correctly and update the paths in `src/config.py` as necessary.

4. **Run the training**:
   To train the CNN model, execute the following command:
   ```
   python src/training/train.py
   ```

5. **Run the application**:
   After training, you can run the application using:
   ```
   python src/app.py
   ```

## Usage

- The application will load the trained model and allow for inference on new images to detect mouse cursor positions.
- You can modify the configuration settings in `src/config.py` to adjust model parameters and training hyperparameters.

## Overview of the CNN Model

The CNN model is designed to effectively learn features from the input images to accurately predict the mouse cursor's position. The architecture is defined in `src/models/cnn_model.py`, where you can also find methods for compiling and summarizing the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.