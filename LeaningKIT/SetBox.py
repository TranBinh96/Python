import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsView,
    QListWidget, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2


class LabelImg(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.image_dir = ""
        self.label_dir = ""
        self.images = []
        self.current_index = 0
        self.current_image = None

    def initUI(self):
        self.setWindowTitle("LabelImg Clone")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QHBoxLayout()
        side_layout = QVBoxLayout()

        # Graphics View for displaying images
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        main_layout.addWidget(self.view)

        # Side Panel
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_selected_image)
        side_layout.addWidget(self.image_list)

        # Buttons
        self.load_image_button = QPushButton("Load Images")
        self.load_image_button.clicked.connect(self.load_images)
        side_layout.addWidget(self.load_image_button)

        self.prev_button = QPushButton("<< Previous")
        self.prev_button.clicked.connect(self.prev_image)
        side_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(self.next_image)
        side_layout.addWidget(self.next_button)

        self.setCentralWidget(QWidget())
        central_layout = QHBoxLayout()
        self.centralWidget().setLayout(central_layout)
        central_layout.addLayout(main_layout)
        central_layout.addLayout(side_layout)

    def load_images(self):
        """
        Tải thư mục ảnh và hiển thị trong danh sách.
        """
        self.image_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not self.image_dir:
            return
        self.images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.image_list.clear()
        self.image_list.addItems(self.images)
        if self.images:
            self.current_index = 0
            self.load_image()

    def load_image(self):
        """
        Tải và hiển thị ảnh hiện tại.
        """
        if not self.images or self.current_index >= len(self.images):
            return

        image_path = os.path.join(self.image_dir, self.images[self.current_index])
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            return

        # Chuyển ảnh sang QImage
        height, width, _ = self.current_image.shape
        qimage = QImage(self.current_image.data, width, height, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)

        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def load_selected_image(self, item):
        """
        Tải ảnh được chọn từ danh sách.
        """
        self.current_index = self.images.index(item.text())
        self.load_image()

    def prev_image(self):
        """
        Chuyển sang ảnh trước đó.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        """
        Chuyển sang ảnh tiếp theo.
        """
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.load_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelImg()
    window.show()
    sys.exit(app.exec_())
