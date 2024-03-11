import os
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys

def find_bone_level(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    bone_level_img = img.copy()
    cv2.drawContours(bone_level_img, contours, -1, (0, 255, 0), 2)

    return bone_level_img

class ImageProcessor(QThread):
    imageProcessed = pyqtSignal(str, 'PyQt_PyObject')

    def __init__(self, files, threshold):
        super().__init__()
        self.files = files
        self.threshold = threshold

    def run(self):
        for file in self.files:
            input_path = os.path.abspath(file)

            img = cv2.imread(input_path)
            if img is None:
                self.imageProcessed.emit(f"Unable to read image: {os.path.basename(input_path)}", None)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            masked_img = cv2.bitwise_and(img, img, mask=mask)

            # Find bone level
            bone_level_img = find_bone_level(input_path)

            self.imageProcessed.emit(os.path.basename(input_path), (masked_img, bone_level_img))

        self.imageProcessed.emit("All images processed.", None)

class ImageMaskingTool(QMainWindow):
    def __init__(self):
        super(ImageMaskingTool, self).__init__()

        self.setWindowTitle("Image Masking Tool")
        self.setGeometry(100, 100, 1200, 600)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.input_label = QLabel("Input Directory:")
        self.layout.addWidget(self.input_label)

        self.input_dir_button = QPushButton("Browse")
        self.input_dir_button.clicked.connect(self.choose_input_dir)
        self.layout.addWidget(self.input_dir_button)

        self.process_button = QPushButton("Process Images")
        self.process_button.clicked.connect(self.process_images)
        self.layout.addWidget(self.process_button)

        self.slider_label = QLabel("Threshold Value:")
        self.layout.addWidget(self.slider_label)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(127)
        self.slider.setEnabled(True)  # Enabled by default
        self.slider.valueChanged.connect(self.update_threshold_label)
        self.layout.addWidget(self.slider)

        self.threshold_label = QLabel("Threshold: 127")
        self.layout.addWidget(self.threshold_label)

        self.threshold_edit = QLineEdit()
        self.threshold_edit.setPlaceholderText("Enter Threshold Value")
        self.threshold_edit.returnPressed.connect(self.update_slider_from_edit)
        self.layout.addWidget(self.threshold_edit)

        self.image_layout = QVBoxLayout()

        self.central_widget.setLayout(self.layout)
        self.layout.addLayout(self.image_layout)

        self.threshold = 127
        self.input_dir = ""

    def choose_input_dir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.input_dir = QFileDialog.getExistingDirectory(self, "Choose Input Directory", options=options)
        self.input_dir_button.setText(self.input_dir)

    def update_threshold_label(self):
        self.threshold = self.slider.value()
        self.threshold_label.setText("Threshold: " + str(self.threshold))
        self.threshold_edit.setText(str(self.threshold))

    def update_slider_from_edit(self):
        try:
            value = int(self.threshold_edit.text())
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            self.threshold = value
            self.slider.setValue(value)
            self.threshold_label.setText("Threshold: " + str(self.threshold))
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid threshold value.")

    def process_images(self):
        if not self.input_dir:
            QMessageBox.critical(self, "Error", "Please select input directory.")
            return

        img_formats = ['*.jpg', '*.jpeg', '*.png']
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", self.input_dir, "Images ({})".format(" ".join(img_formats)))
        if not files:
            QMessageBox.critical(self, "Error", "No images selected.")
            return

        self.process_button.setEnabled(False)  # Disable process button
        self.slider.setEnabled(False)  # Disable slider during processing

        self.processor = ImageProcessor(files, self.threshold)
        self.processor.imageProcessed.connect(self.update_image)
        self.processor.finished.connect(self.enable_controls)
        self.processor.start()

    def enable_controls(self):
        self.process_button.setEnabled(True)  # Re-enable process button
        self.slider.setEnabled(True)  # Re-enable slider after processing

    def update_image(self, filename, images):
        if images is not None:
            masked_img, bone_level_img = images

            # Resize images
            masked_img_resized = cv2.resize(masked_img, (400, 300))  # Adjust dimensions as needed
            bone_level_img_resized = cv2.resize(bone_level_img, (400, 300))  # Adjust dimensions as needed

            # Convert resized images to QPixmap
            masked_pixmap = self.convert_cv_to_qpixmap(masked_img_resized)
            bone_level_pixmap = self.convert_cv_to_qpixmap(bone_level_img_resized)

            # Create QLabel widgets for displaying images
            masked_label = QLabel()
            masked_label.setPixmap(masked_pixmap)

            bone_level_label = QLabel()
            bone_level_label.setPixmap(bone_level_pixmap)

            # Create a layout for displaying both images side by side
            h_layout = QHBoxLayout()
            h_layout.addWidget(masked_label)
            h_layout.addWidget(bone_level_label)

            # Add the layout to the image layout of the window
            self.image_layout.addLayout(h_layout)
        else:
            QMessageBox.warning(self, "Warning", filename)

    def convert_cv_to_qpixmap(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        return QtGui.QPixmap.fromImage(q_img)

def main():
    app = QApplication(sys.argv)
    window = ImageMaskingTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
