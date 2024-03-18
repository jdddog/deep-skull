# Copyright 2021 James Diprose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
from operator import attrgetter
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import measure


def transform_image(image: np.ndarray, size: Tuple[int, int] = (512, 512)):
    # Move Z axis from 2 to 0
    image = np.moveaxis(image, source=2, destination=0)

    # Pad to square to maintain aspect ratio and resize to desired size
    slice_shape = image.shape[1:]
    max_len = max(slice_shape)
    row_change = max_len - slice_shape[0]
    col_change = max_len - slice_shape[1]
    if slice_shape != size:
        slices_padded = np.pad(image, [(0, 0), (0, row_change), (0, col_change)], mode="edge")

        slices = []
        for slice in slices_padded:
            slices.append(np.array(Image.fromarray(slice).resize(size)))
        image = np.array(slices)

    # Rotate 90 degrees to the right
    image = np.rot90(image, k=1, axes=(1, 2))

    return image


def de_transform_mask(image: np.ndarray, mask: np.ndarray, size: Tuple[int, int] = (512, 512)):
    # Rotate 45 degrees to the left
    mask = np.rot90(mask, k=3, axes=(1, 2))

    # Reverse resize and pad to square to maintain aspect ratio and resize to desired size
    input_row = image.shape[0]
    input_col = image.shape[1]
    slice_shape = image.shape[:-1]
    max_len = max(slice_shape)
    if slice_shape != size:
        slices = []
        for slice in mask:
            slices.append(np.array(Image.fromarray(slice).resize((max_len, max_len), Image.NEAREST)))

        mask = np.array(slices)
        mask = mask[:, :input_row, :input_col]

    # Move Z axis from 0 to 2
    mask = np.moveaxis(mask, source=0, destination=2)

    return mask


def compute_contiguous_mask(*, mask: np.ndarray):
    """Get the largest contiguous mask.

    :param mask: a mask.
    :return: the largest contiguous mask and it's bounding box.
    """

    # Get largest contiguous mask
    labels = measure.label(mask, background=0)
    regions = measure.regionprops(labels)
    largest_region = max(regions, key=attrgetter("area"))
    bbox = largest_region.bbox  # The bounding box values equal: minz, minr, minc, maxz, maxr, maxc
    label = largest_region.label
    mask = np.array(labels == label, dtype=np.int8)
    return mask, bbox


def fill_mask_holes(*, mask: np.ndarray):
    """Fill the holes in a mask.

    :param mask: a mask.
    :return: the mask with holes filled.
    """

    filled = []
    for img in mask:
        # Compute labels of image
        labels = measure.label(img, background=1)

        # Make a mask for each corner in case they have different labels
        # numpy indexes r,c : top left, top right, bottom left and bottom right
        c_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        c_masks = []
        for label in c_labels:
            c_masks.append(label == labels)
        c_masks = np.array(c_masks)

        # Combine masks
        filled_mask = reduce(np.logical_or, c_masks)
        filled_mask = np.logical_not(filled_mask)
        filled.append(filled_mask)

    filled = np.array(filled, dtype=np.int8)
    return filled
