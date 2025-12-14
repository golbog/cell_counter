from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import feature, segmentation

from model.resunetpp import ResUNetPP
from model.c_resunet import CResUnet
from model.fpn import FPN
from data_handler.weight_masks import cleanup_mask
from utils import PathLike


class CellCounter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, threshold: float = 0.5):
        ...

    def filter(self, segmented_image: np.ndarray, threshold: float = 0.5, foot=40, shed=True):
        filtering = cleanup_mask(segmented_image, threshold=threshold, small_obj_size=foot, small_hole_size=foot)
        if shed:
            filtering = np.squeeze(filtering).astype('uint8')  # make sure it's in expected shape and type
            distance = ndimage.distance_transform_edt(filtering)
            markers = np.zeros(distance.shape, dtype=bool)
            local_maxi = feature.peak_local_max(distance, footprint=np.ones((foot, foot)),
                                                exclude_border=False, labels=filtering)
            markers[tuple(local_maxi.T)] = True
            markers, _ = ndimage.label(markers)
            labels = segmentation.watershed(-distance, markers, mask=filtering,
                                            compactness=1, watershed_line=True)

            return (labels > 0.5).astype("uint8")
        return filtering

    def count_cells_from_filtered(self, filtered_image):
        ...

    def count_cells_from_raw(self, image):
        ...

    @abstractmethod
    def save(self, filename):
        ...

    @staticmethod
    @abstractmethod
    def load(filename):
        ...


class PytorchCellCounter(CellCounter):
    import torch as t

    def __init__(self, model, weight_filename: Optional[PathLike] = None, device=None, size_mult=1):
        self.model = model
        self.size_mult = size_mult
        if weight_filename is not None:
            self.model.load_state_dict(self.t.load(weight_filename, map_location=self._get_device(device), weights_only=True))

        self.device = self._get_device(device)

    def segment(self, image: np.ndarray):
        # handle shape
        reduce_shape = False
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]
            reduce_shape = True

        # handle dimension
        if self.size_mult != 1:
            max_allowed_shape = np.array(image.shape[-2:]) // self.size_mult * self.size_mult
            # crop TODO: center crop
            image = image[:, :, :max_allowed_shape[0], :max_allowed_shape[1]]

        with self.t.no_grad():
            res = self.model(self.t.from_numpy(image).to(self.device).float())
        res = res.cpu().numpy()
        if reduce_shape:
            res = res[0]
        return res

    def save(self, name):
        # TODO: abstract in base class
        checkpoint = {
            'state_dict': self.model.state_dict(),
            '_model_name_': self.model.__class__.__name__,
            '_input_num_': self.model.num_channels,  # TODO: abstract or other way to get it?
            '_filters_': self.model.filters
        }
        self.t.save(checkpoint, name)

    @staticmethod
    def _get_device(device=None):
        if device is not None:
            return device

        return "cuda" if PytorchCellCounter.t.cuda.is_available() else "cpu"

    @staticmethod
    def load(filename, device=None):
        device = PytorchCellCounter._get_device(device)
        checkpoint = PytorchCellCounter.t.load(filename, map_location=PytorchCellCounter.t.device(device))
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_name = checkpoint['_model_name_']
            input_num = checkpoint['_input_num_']
            filters = checkpoint['_filters_']
        else:
            state_dict = checkpoint
            model_name = state_dict['_model_name_']
            input_num = state_dict['_input_num_']
            filters = state_dict['_filters_']
            del state_dict['_model_name_']  # TODO: maybe delete is not needed
            del state_dict['_input_num_']
            del state_dict['_filters_']

        if model_name == ResUNetPP.__name__:
            model = ResUNetPP(input_num, filters)
        elif model_name == CResUnet.__name__:
            model = CResUnet(input_num, filters)
        elif model_name == FPN.__name__:
            model = FPN(input_num, filters)
        else:
            print(f'Trying to load unknown model with name: {model_name}')
            return None

        model.load_state_dict(state_dict)
        model.to(PytorchCellCounter._get_device(device))
        model.eval()
        return PytorchCellCounter(model, size_mult=model.get_size_mult())
