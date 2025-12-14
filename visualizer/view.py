import random
import sys

# We need to import torch before PyQt5 to avoid DLL loading issues (WinError 1114)
# This is a known issue with conflicting DLLs (like libiomp5md.dll)
import torch

import numpy as np
import qdarkstyle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QListView, QSlider, QSizePolicy, QFileDialog
from PyQt5 import uic, QtGui, QtCore

from model.cell_counter import PytorchCellCounter
from visualizer.qmodel import ImgListModel, ImageThread

DEFAULT_DATASET_PATH = "../data/train/"


class UI(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()

        # load ui file
        uic.loadUi("window.ui", self)

        # define widgets
        self.button_random = self.findChild(QPushButton, "random_img_button")
        self.button_random.clicked.connect(self.random_button)

        self.label_img_source = self.findChild(QLabel, "source_img_display")
        self.label_mask_source = self.findChild(QLabel, "mask_img_display")
        self.label_weight_source = self.findChild(QLabel, "weight_img_display")
        self.label_prediction_result = self.findChild(QLabel, "prediction_img_display")
        self.label_filtered_result = self.findChild(QLabel, "filtered_prediction_img_display")

        # set size policies to allow labels to expand
        for label in [self.label_img_source, self.label_mask_source, self.label_weight_source,
                      self.label_prediction_result, self.label_filtered_result]:
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumSize(1, 1)

        # store original pixmaps for rescaling
        self.original_source_pixmap = None
        self.original_mask_pixmap = None
        self.original_weight_pixmap = None
        self.original_prediction_pixmap = None
        self.original_filtered_pixmap = None

        self.threshold_slider = self.findChild(QSlider, "threshold_slider")
        self.threshold_slider.valueChanged.connect(self.filter_parameters_changed)

        self.list_view_imgs: QListView = self.findChild(QListView, "img_list")
        self.imgs_model = ImgListModel()
        self.imgs_model.update_list(DEFAULT_DATASET_PATH)
        self.list_view_imgs.setModel(self.imgs_model)
        self.list_view_imgs.selectionModel().currentChanged.connect(self.img_selected)

        # threads
        if model_path is None:
            model_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pth)")

        self.cell_counter = PytorchCellCounter.load(model_path)
        self.seg_thread = ImageThread()
        self.filt_thread = ImageThread()

        self.seg_thread.set_callable(self.cell_counter.segment)
        self.filt_thread.set_callable(self.cell_counter.filter)
        self.filt_thread.set_parameters(threshold=self.threshold_slider.value() / 100.)

        self.seg_thread.start()
        self.filt_thread.start()

        self.seg_thread.image_done.connect(self.segmentation_done)
        self.filt_thread.image_done.connect(self.filtering_done)

        # show app
        self.show()

    def img_selected(self, selection: QtGui.QStandardItem, before: QtGui.QStandardItem):
        self.display_image(selection.data())

    def filter_parameters_changed(self):
        self.filt_thread.set_parameters(threshold=self.threshold_slider.value() / 100.)
        self.filt_thread.new_run()

    def segmentation_done(self):
        self.original_prediction_pixmap = self.nparray_to_full_pixelmap(self.seg_thread.output_image)
        self.update_prediction_display()

        self.filt_thread.set_image(self.seg_thread.output_image)
        self.filt_thread.new_run()

    def filtering_done(self):
        self.original_filtered_pixmap = self.nparray_to_full_pixelmap(self.filt_thread.output_image)
        self.update_filtered_display()

    def update_prediction_display(self):
        if self.original_prediction_pixmap:
            scaled = self.original_prediction_pixmap.scaled(
                self.label_prediction_result.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.label_prediction_result.setPixmap(scaled)

    def update_filtered_display(self):
        if self.original_filtered_pixmap:
            scaled = self.original_filtered_pixmap.scaled(
                self.label_filtered_result.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.label_filtered_result.setPixmap(scaled)

    def update_source_display(self):
        if self.original_source_pixmap:
            scaled = self.original_source_pixmap.scaled(
                self.label_img_source.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.label_img_source.setPixmap(scaled)

    def update_mask_display(self):
        if self.original_mask_pixmap:
            scaled = self.original_mask_pixmap.scaled(
                self.label_mask_source.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.label_mask_source.setPixmap(scaled)

    def update_weight_display(self):
        if self.original_weight_pixmap:
            scaled = self.original_weight_pixmap.scaled(
                self.label_weight_source.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.label_weight_source.setPixmap(scaled)

    def display_image(self, name):
        source, mask, weights = self.imgs_model.get_imgs_for(name)

        self.original_source_pixmap = self.nparray_to_full_pixelmap(source)
        self.original_mask_pixmap = self.nparray_to_full_pixelmap(mask)
        self.original_weight_pixmap = self.nparray_to_full_pixelmap(weights)

        self.update_source_display()
        self.update_mask_display()
        self.update_weight_display()

        self.label_prediction_result.setText("Loading...")
        self.label_filtered_result.setText("Loading...")

        self.seg_thread.set_image(source)
        self.seg_thread.new_run()

    def random_button(self):
        random_row = random.randint(0, self.imgs_model.rowCount() - 1)
        self.list_view_imgs.selectionModel().setCurrentIndex(self.imgs_model.index(random_row, 0), QtCore.QItemSelectionModel.ClearAndSelect)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_source_display()
        self.update_mask_display()
        self.update_weight_display()
        self.update_prediction_display()
        self.update_filtered_display()

    @staticmethod
    def nparray_to_full_pixelmap(array, scale_pixel=True):
        """Convert numpy array to QPixmap at original size."""
        if len(array.shape) == 3:
            channel, height, width = array.shape
        else:
            height, width = array.shape[:2]
            channel = 1

        if scale_pixel:
            array = array * 255 / np.max(array)
        array = array.astype(np.uint8)

        format = QImage.Format_Grayscale8 if channel == 1 else QImage.Format_RGB888
        bytes_per_line = channel * width
        img = QImage(array, width, height, bytes_per_line, format)
        pixmap = QPixmap.fromImage(img)
        return pixmap

    @staticmethod
    def nparray_to_pixelmap(array, w, h, scale_pixel=True):
        if len(array.shape) == 3:
            channel, height, width = array.shape
        else:
            height, width = array.shape[:2]
            channel = 1

        if scale_pixel:
            array = array * 255 / np.max(array)
        array = array.astype(np.uint8)

        format = QImage.Format_Grayscale8 if channel == 1 else QImage.Format_RGB888
        bytes_per_line = channel * width
        img = QImage(array, width, height, bytes_per_line, format)
        pixmap = QPixmap.fromImage(img).scaled(w, h, QtCore.Qt.KeepAspectRatio)
        return pixmap


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui_window = UI()
    app.exec()
