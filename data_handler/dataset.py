import numpy as np
import torch
import torchvision
from typing import List, Optional, Dict

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from utils import PathLike
from .augmentor import Augmentor
from .cc_image import CCImage


class CCDataset(Dataset):
    def __init__(self, train: bool, imgs_root: PathLike, masks_root: Optional[PathLike] = None, weights_root: Optional[PathLike] = None,
                 transform: Optional[Augmentor] = None, subset_names: Optional[List[str]] = None,
                 data: Optional[Dict[str, CCImage]] = None) -> None:

        if train and masks_root is None and data is None:
            raise RuntimeError("Cannot create train dataset without provided masks folder")

        self._train = train
        self._imgs_root = Path(imgs_root)
        self._masks_root = Path(masks_root) if masks_root is not None else None
        self._weights_root = Path(weights_root) if weights_root is not None else None

        self._transform = transform

        if data is not None:
            self._imgs = data
        else:
            self._imgs = self.load_dataset_structure(self._imgs_root, self._masks_root, self._weights_root)

        if subset_names is not None:
            self._imgs = {k: v for k, v in self._imgs.items() if k in subset_names}
        self._imgs_vals = list(self._imgs.values())

    @staticmethod
    def load_dataset_structure(imgs_root, masks_root, weights_root) -> Dict[str, CCImage]:
        res = dict()
        # TODO: this could be saved in a file, so it doesn't need to be run everytime
        for path in imgs_root.glob("*"):
            if path.is_dir() and (masks_root / path.stem).exists():
                res.update(CCDataset.load_dataset_structure(path, masks_root / path.stem if masks_root is not None else None,
                                                             weights_root / path.stem if weights_root is not None else None))
            else:  # is a file
                mask_path = None
                if masks_root is not None:
                    mask_path = CCDataset.filename_in_folder(path.stem, masks_root)
                    if mask_path is None:
                        continue

                weight_mask = None
                if weights_root is not None:
                    weight_mask = CCDataset.filename_in_folder(path.stem, weights_root)
                    if weight_mask is None:
                        continue

                res[path.stem] = CCImage(path, mask_path, weight_mask)

        return res

    @staticmethod
    def filename_in_folder(filename, folder):
        candidates = list(folder.glob(f"{filename}.*"))
        if candidates is not None and len(candidates) == 1:
            return candidates[0]

        return None

    def file_names(self):
        return list(self._imgs.keys())

    def get_item(self, name):
        cc_img = self._imgs[name]
        return self._item_to_return(cc_img)

    def __len__(self):
        return len(self._imgs)

    def _item_to_return(self, cc_img):
        img, mask, weights = cc_img.as_tuple()
        if self._transform is not None:
            img, (mask, weights) = self._transform(img, masks=(mask, weights, ))

        img = np.rollaxis(img, 2, 0)
        mask = mask[np.newaxis, ...]
        weights = weights[np.newaxis, ...]
        return img, mask, weights

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cc_img = self._imgs_vals[idx]

        return self._item_to_return(cc_img)
