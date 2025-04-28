class DatasetLoader:
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir
        self.label_file = label_file
        self.images = []
        self.labels = []

    def load_dataset(self):
        self.images, self.labels = self._load_images_and_labels()

    def _load_images_and_labels(self):
        # Implement logic to load images and labels from the specified directory and file
        pass

    def get_images(self):
        return self.images

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.images)