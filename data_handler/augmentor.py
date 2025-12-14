from abc import ABC, abstractmethod
from typing import Tuple, Optional

import albumentations as A
import numpy as np


class Augmentor(ABC):
    """
    Abstract augmentor interface for augmenting cell images and their masks.
    """
    @abstractmethod
    def __call__(self, img: np.ndarray, masks: Tuple[np.ndarray, ...]):
        """
        Transforms the input data.

        :param img: input image as numpy array
        :param masks: tuple of masks (ground truth, weights, etc.)
        :return: transformed image and masks
        """
        ...


class AlbumentationAugmentor(Augmentor):
    """
    Augmentor for cell counting training using albumentations library.
    Includes augmentations suitable for microscopy cell images.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (1024, 1024),
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        p_elastic: float = 0.3,
        p_noise: float = 0.3,
        p_blur: float = 0.2,
        p_brightness_contrast: float = 0.3,
        p_gamma: float = 0.2,
    ):
        """
        Initialize augmentor with configurable probabilities.

        :param crop_size: (height, width) for random crop
        :param p_flip: probability for horizontal/vertical flips
        :param p_rotate: probability for rotation
        :param p_elastic: probability for elastic deformation
        :param p_noise: probability for noise augmentations
        :param p_blur: probability for blur augmentations
        :param p_brightness_contrast: probability for brightness/contrast adjustment
        :param p_gamma: probability for gamma correction
        """
        self.crop_size = crop_size
        self.transform = A.Compose([
            # Spatial transforms - safe for segmentation masks
            A.RandomCrop(height=crop_size[0], width=crop_size[1]),
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_rotate),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-45, 45),
                p=0.5,
            ),

            # Elastic deformation - simulates biological variability
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=p_elastic
            ),

            # Intensity transforms - only applied to image, not masks
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=p_noise),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=p_blur),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=p_brightness_contrast
            ),

            A.RandomGamma(gamma_limit=(80, 120), p=p_gamma),

            # CLAHE - useful for microscopy images with varying illumination
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        ], is_check_shapes=False)  # Disable shape check, we handle it manually

    def __call__(self, img: np.ndarray, masks: Tuple[np.ndarray, ...]):
        # Ensure masks match image dimensions (H, W)
        img_h, img_w = img.shape[:2]
        resized_masks = []
        for mask in masks:
            if mask.shape[0] != img_h or mask.shape[1] != img_w:
                # Resize mask to match image using nearest neighbor (preserves binary values)
                resized = A.Resize(height=img_h, width=img_w, interpolation=0)(image=mask)["image"]
                resized_masks.append(resized)
            else:
                resized_masks.append(mask)
        
        transformed = self.transform(image=img, masks=resized_masks)
        return transformed["image"], transformed["masks"]


class LightAugmentor(Augmentor):
    """
    Lighter augmentation for faster training or fine-tuning.
    Only includes basic spatial transforms.
    """

    def __init__(self, crop_size: Tuple[int, int] = (1024, 1024)):
        self.transform = A.Compose([
            A.RandomCrop(height=crop_size[0], width=crop_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], is_check_shapes=False)

    def __call__(self, img: np.ndarray, masks: Tuple[np.ndarray, ...]):
        # Ensure masks match image dimensions (H, W)
        img_h, img_w = img.shape[:2]
        resized_masks = []
        for mask in masks:
            if mask.shape[0] != img_h or mask.shape[1] != img_w:
                resized = A.Resize(height=img_h, width=img_w, interpolation=0)(image=mask)["image"]
                resized_masks.append(resized)
            else:
                resized_masks.append(mask)
        
        transformed = self.transform(image=img, masks=resized_masks)
        return transformed["image"], transformed["masks"]


class Cropper(Augmentor):
    """
    Simple center crop for validation/evaluation.
    Ensures consistent image sizes without random augmentation.
    """

    def __init__(self, crop_size: Tuple[int, int] = (1024, 1024)):
        self.transform = A.CenterCrop(height=crop_size[0], width=crop_size[1])

    def __call__(self, img: np.ndarray, masks: Tuple[np.ndarray, ...]):
        # Ensure masks match image dimensions (H, W)
        img_h, img_w = img.shape[:2]
        resized_masks = []
        for mask in masks:
            if mask.shape[0] != img_h or mask.shape[1] != img_w:
                resized = A.Resize(height=img_h, width=img_w, interpolation=0)(image=mask)["image"]
                resized_masks.append(resized)
            else:
                resized_masks.append(mask)
        
        transformed = self.transform(image=img, masks=resized_masks)
        return transformed["image"], transformed["masks"]


class ResizeAugmentor(Augmentor):
    """
    Resize images to a fixed size. Useful for inference on variable-sized inputs.
    """

    def __init__(self, size: Tuple[int, int] = (1024, 1024)):
        self.transform = A.Resize(height=size[0], width=size[1])

    def __call__(self, img: np.ndarray, masks: Tuple[np.ndarray, ...]):
        transformed = self.transform(image=img, masks=masks)
        return transformed["image"], transformed["masks"]
