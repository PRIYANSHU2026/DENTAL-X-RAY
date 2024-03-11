import os
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit, QHBoxLayout  # Import QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys

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

            self.imageProcessed.emit(os.path.basename(input_path), masked_img)

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

    def update_image(self, filename, img):
        if img is not None:
            # Resize the processed image
            img_height, img_width, _ = img.shape
            max_height = 300
            max_width = 400
            scaling_factor = min(max_width / img_width, max_height / img_height)
            resized_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            q_img = QtGui.QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0], resized_img.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap = QtGui.QPixmap.fromImage(q_img)
            processed_label = QLabel()
            processed_label.setPixmap(pixmap)

            # Load and resize the original X-ray image
            original_img_path = os.path.join(self.input_dir, filename)
            original_img = cv2.imread(original_img_path)
            resized_original_img = cv2.resize(original_img, (int(original_img.shape[1] * scaling_factor), int(original_img.shape[0] * scaling_factor)))

            original_q_img = QtGui.QImage(resized_original_img.data, resized_original_img.shape[1], resized_original_img.shape[0], resized_original_img.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
            original_pixmap = QtGui.QPixmap.fromImage(original_q_img)
            original_label = QLabel()
            original_label.setPixmap(original_pixmap)

            # Create a layout for displaying both images side by side
            h_layout = QHBoxLayout()
            h_layout.addWidget(original_label)
            h_layout.addWidget(processed_label)

            # Add the layout to the image layout of the window
            self.image_layout.addLayout(h_layout)
        else:
            QMessageBox.warning(self, "Warning", filename)

def main():
    app = QApplication(sys.argv)
    window = ImageMaskingTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
