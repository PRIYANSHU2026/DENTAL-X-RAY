import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = models.vgg16(pretrained=True)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match VGG input size
    transforms.ToTensor(),
])

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Image Viewer'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a button in the window
        self.button = QPushButton('Open image', self)
        self.button.clicked.connect(self.open_image)

        # Create a label which will show the image
        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setScaledContents(True)

        # Add button and label to the layout
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)

    def open_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            # Load and preprocess the image
            image = Image.open(fileName).convert('RGB')
            image = transform(image).unsqueeze(0)

            # Pass image through the model
            with torch.no_grad():
                conv_output_image = model.features(image)

            # Normalize the feature maps
            def normalize_feature_maps(feature_maps):
                # Normalize each channel independently
                for i in range(feature_maps.shape[-1]):
                    # Get the maximum and minimum values in each channel
                    max_val = np.max(feature_maps[:, :, i])
                    min_val = np.min(feature_maps[:, :, i])
                    # Check if max_val and min_val are equal to handle division by zero
                    if max_val == min_val:
                        feature_maps[:, :, i] = 0
                    else:
                        feature_maps[:, :, i] = (feature_maps[:, :, i] - min_val) / (max_val - min_val)
                return feature_maps

            # Normalize the conv_output_image
            conv_output_image = normalize_feature_maps(conv_output_image)

            # Convert the tensor to numpy array
            overlay_image = conv_output_image.squeeze(0).permute(1, 2, 0).numpy()

            # Convert the numpy array to QImage
            qimage = QImage(overlay_image.data, overlay_image.shape[1], overlay_image.shape[0], QImage.Format_RGB888)

            # Convert the QImage to QPixmap and show it on the label
            pixmap = QPixmap.fromImage(qimage)
            self.label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())
