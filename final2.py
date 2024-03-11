import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch

# Load the pre-trained model
model = models.vgg16(pretrained=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match VGG input size
    transforms.ToTensor(),
])

# Function to select a random image from the 'Y' directory
def get_random_label_image(y_dir):
    # Get a list of all image file paths in the 'Y' directory
    image_paths = [os.path.join(y_dir, filename) for filename in os.listdir(y_dir)
                   if filename.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Check if there are any images in the directory
    if not image_paths:
        raise ValueError("No images found in the 'Y' directory")

    # Select a random image path from the list
    random_image_path = random.choice(image_paths)
    return random_image_path

# Normalize the feature maps
def normalize_feature_maps(feature_maps):
    for i in range(feature_maps.shape[-1]):
        max_val = np.max (feature_maps[:, :, i])
        min_val = np.min(feature_maps[:, :, i])
        if max_val == min_val:
            feature_maps[:, :, i] = 0
        else:
            feature_maps[:, :, i] = (feature_maps[:, :, i] - min_val) / (max_val - min_val)
    return feature_maps

# Create a simple PyQt5 GUI
class ImageSelector(QWidget):
    def __init__(self, y_dir):
        super().__init__()
        self.y_dir = y_dir
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Selector')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setText("No Image Selected")
        layout.addWidget(self.image_label)

        select_button = QPushButton('Select Image', self)
        select_button.clicked.connect(self.select_image)
        layout.addWidget(select_button)

        process_button = QPushButton('Process Image', self)
        process_button.clicked.connect(self.process_image)
        layout.addWidget(process_button)

        self.setLayout(layout)

    def select_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.selectFile("")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image_label.setPixmap(QPixmap(file_path).scaled(200, 200))
            self.image_path = file_path

    def process_image(self):
        image = Image.open(self.image_path).convert('RGB')
        label_path = get_random_label_image(self.y_dir)  # Use self.y_dir here
        label = Image.open(label_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        label = transform(label).unsqueeze(0)

        with torch.no_grad():
            conv_output_image = model.features(image)
            conv_output_label = model.features(label)

        overlay_image = (1 - 0.5) * image.squeeze(0).permute(1, 2, 0).numpy() + 0.5 * label.squeeze(0).permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.title('prediction label of bone loss')
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    y_dir = "/Users/shikarichacha/Desktop/CNN/Y"  # Path to the 'Y' directory
    selector = ImageSelector(y_dir)
    selector.show()
    sys.exit(app.exec_())
