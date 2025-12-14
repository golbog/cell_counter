import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import pyqtSignal, QThread

from data_handler.dataset import CCDataset
from utils import PathLike


class ImgListModel(QStandardItemModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._root = None
        self._dataset = None
        self._names = None

    def update_list(self, path: PathLike):
        if path is not None:
            self._root = Path(path)
        self._dataset = CCDataset(False, self._root / "images", self._root / "masks", self._root / "weights")
        self._names = self._dataset.file_names()

        for i in self._names:
            item = QStandardItem(i)
            self.appendRow(item)

    def get_imgs_for(self, name):
        return self._dataset.get_item(name)


class ImageThread(QThread):
    image_done = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_image = None
        self.output_image = None
        self.parameters = None

        self._stop = False
        self._new_run = False

    def set_callable(self, model: Callable, **kwargs):
        self.model = model
        self.parameters = kwargs

    def set_parameters(self, **kwargs):
        self.parameters = kwargs

    def set_image(self, image: np.ndarray):
        self.input_image = image
        self.output_image = None

    def stop(self):
        self._stop = True

    def new_run(self):
        self._new_run = True

    def run(self) -> None:
        while not self._stop:
            if not self._new_run:
                self.msleep(250)  # TODO: parametrize
                continue
            if self.input_image is None or self.model is None:
                self._new_run = False
                warnings.warn("First set the image and mode")
                continue
            self._new_run = False  # TODO: queue

            self.output_image = self.model(self.input_image, **self.parameters)
            self.image_done.emit()
