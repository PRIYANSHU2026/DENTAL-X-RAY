import os
import random
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PIL import Image
from torchvision import models, transforms
import torch
import numpy as np

class ImageLabeler(QMainWindow):
    def __init__(self):
        super(ImageLabeler, self).__init__()

        self.setWindowTitle("Image Labeler")
        self.setGeometry(100, 100, 1200, 600)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()

        self.input_label = QLabel("Input Image:")
        self.layout.addWidget(self.input_label, 0, 0)

        self.input_image_label = QLabel()
        self.layout.addWidget(self.input_image_label, 1, 0, 1, 2)

        self.choose_button = QPushButton("Choose Input Image")
        self.choose_button.clicked.connect(self.choose_input_image)
        self.layout.addWidget(self.choose_button, 2, 0, 1, 2)

        self.label_label = QLabel("Label Image:")
        self.layout.addWidget(self.label_label, 0, 2)

        self.label_image_label = QLabel()
        self.layout.addWidget(self.label_image_label, 1, 2, 1, 2)

        self.random_button = QPushButton("Random Label Image")
        self.random_button.clicked.connect(self.load_random_label_image)
        self.layout.addWidget(self.random_button, 2, 2, 1, 2)

        self.central_widget.setLayout(self.layout)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.model = models.vgg16(pretrained=True)

    def choose_input_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_path, _ = QFileDialog.getOpenFileName(self, "Choose Input Image", "", "Images (*.png *.jpg *.jpeg)")
        if image_path:
            self.load_images(image_path, self.label_path)

    def load_random_label_image(self):
        label_dir = "/Users/shikarichacha/Desktop/CNN/Y"
        label_files = [f"{i}.jpg" for i in range(1, 51)]  # List of label files from 1.jpg to 50.jpg
        random_label_file = random.choice(label_files)
        label_path = os.path.join(label_dir, random_label_file)

        self.load_images(self.image_path, label_path)

    def load_images(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        # Resize the label image to match the size of the original image
        label = label.resize(image.size)

        input_image_tensor = self.transform(image).unsqueeze(0)
        label_image_tensor = self.transform(label).unsqueeze(0)

        with torch.no_grad():
            input_conv_output = self.model.features(input_image_tensor)
            label_conv_output = self.model.features(label_image_tensor)

        overlay_image = self.overlay_images(image, label, alpha=0.5)

        self.display_image(input_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), self.input_image_label)
        self.display_image(label_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), self.label_image_label)
        self.display_image(overlay_image, self.label_image_label)

    def overlay_images(self, image, label, alpha=0.5):
        overlay_image = (1 - alpha) * np.array(image) + alpha * np.array(label)
        return overlay_image

    def display_image(self, image, label_widget):
        q_image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        label_widget.setPixmap(pixmap)

def main():
    app = QApplication([])
    window = ImageLabeler()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()

