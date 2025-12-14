from pathlib import Path

import numpy as np
import cv2 as cv
from skimage import morphology


def cleanup_mask(img: np.ndarray, threshold: float = 150, small_obj_size: int =25, small_hole_size: int = 10) -> np.ndarray:
    img = img > threshold
    img = morphology.remove_small_objects(img, small_obj_size)
    img = morphology.remove_small_holes(img, small_hole_size)

    return (img * 255).astype(np.uint8)


def weight_from_mask_cell_border(mask: np.ndarray, weight_edge=1, edge_thickness=18) -> np.ndarray:
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    smoothed_contours = [cv.approxPolyDP(contour, 0, True) for contour in contours]

    new_img = np.zeros(mask.shape, np.float32)
    buffer_img = np.zeros(mask.shape, dtype=np.float32)

    for contour in smoothed_contours:
        cv.drawContours(buffer_img, [contour], -1, color=(weight_edge,), thickness=edge_thickness)
        new_img += buffer_img
        buffer_img.fill(0)

    # only get the overlaps
    new_img[new_img <= 1] = 0

    return new_img


def min_contour_radius(contour, center):
    # TODO: optimize
    dists = list()
    for point in contour:
        dists.append(np.linalg.norm(point[0] - center))

    return np.min(dists)


def weight_from_mask_centers(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    smoothed_contours = [cv.approxPolyDP(contour, 0, True) for contour in contours]

    new_img = np.zeros(mask.shape, np.float32)

    for contour in smoothed_contours:
        moments = cv.moments(contour)
        if moments["m00"] == 0:
            continue  # avoid division by zero
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        min_radius = min_contour_radius(contour, np.array([center_x, center_y]))
        # draw filled contour
        cv.drawContours(new_img, [contour], -1, (1, ), -1)
        # cut out the edge of the contour, so only the more central part remains
        cv.drawContours(new_img, [contour], -1, (0, ), int(min_radius))

        # draw circle in the middle of contour, to make sure even small contours have notable centers
        # cv.circle(new_img, (center_x, center_y), 8, (1, ), thickness=-1)

    return new_img


# TODO: try this: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_cdt.html
if __name__ == '__main__':
    # masks_path, weights_path = sys.argv[:2]
    masks_path = "data/train/masks"
    weights_path = "data/train/weights"
    masks_path = Path(masks_path)
    weights_path = Path(weights_path)

    for mask_path in masks_path.glob("*.png"):
        mask_orig = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        mask = cleanup_mask(mask_orig)
        border_overlap_weights = weight_from_mask_cell_border(mask)
        center_weights = weight_from_mask_centers(mask)

        weights = border_overlap_weights + center_weights
        weights += 1

        # blur mask a bit more for soft edges
        weights = cv.GaussianBlur(weights, (5, 5), 0)

        cv.imwrite(str(mask_path), mask)
        cv.imwrite(str(weights_path / mask_path.name), weights)
