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

import logging
import os

import ray
import tensorflow as tf
from matplotlib import pyplot as plt
from monai.transforms import AddChannel, LoadImage, SaveImage, ScaleIntensityRange
from monai.visualize import matshow3d

from deep_skull.masks import compute_contiguous_mask, de_transform_mask, fill_mask_holes, transform_image
from deep_skull.model import BrainExtractorModel


def set_gpu_memory_growth(enable: bool):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(f"GPU: {gpu}")
                tf.config.experimental.set_memory_growth(gpu, enable)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logging.error(f"Memory growth must be set before GPUs have been initialized: {str(e)}")


@ray.remote(num_gpus=1)
class BrainExtractorActor:
    def __init__(
        self, local_mode, gpu: int, batch_size: int, debug: bool = False, every_n: int = 5, ct_add_brightness: int = 20
    ):
        """

        :param local_mode:
        :param gpu:
        :param batch_size:
        :param debug:
        :param every_n:
        :param add_brightness:
        """

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

        self.device = f"gpu:{gpu}"
        self.debug = debug
        self.every_n = every_n
        self.ct_add_brightness = ct_add_brightness

        if not local_mode:
            logging.info("Init Tensorflow session")
            set_gpu_memory_growth(True)
            logging.info("Tensorflow session initiated")

        print("start")
        with tf.device(self.device):
            self.brain_extractor = BrainExtractorModel(batch_size=batch_size)
        print("end")

    def extract_brain(self, scan_path: str, mask_path: str, scan_type: str):
        """

        :param scan_path:
        :param mask_path:
        :param scan_type:
        :return:
        """

        # Make metadata
        filename_or_obj = os.path.basename(mask_path)

        # Load image
        logging.info(f"Processing scan, scan_path: {scan_path}, mask_path: {mask_path}.")
        image, meta = LoadImage()(scan_path)

        # Transform image
        transformed = transform_image(image)

        # Add a constant to CT scans as it makes the masks come out more reliably
        if scan_type == "ax_CT":
            transformed = transformed + self.ct_add_brightness

        # Visualise transformed scan
        if self.debug:
            # Print volume
            volume = (ScaleIntensityRange(a_min=0, a_max=100, b_min=0.0, b_max=1.0, clip=True)(transformed),)
            matshow3d(title=filename_or_obj, volume=volume, every_n=self.every_n, show=True, cmap="gray")

            # Print middle image
            middle = int(len(transformed) / 2)
            plt.imshow(transformed[middle])
            plt.show()

        # Get brain masks
        with tf.device(self.device):
            logging.info(f"Using device {self.device}")
            mask = self.brain_extractor.predict(transformed)

        # For CT Angiograms, this cuts out incorrect mask labels from below the neck
        # For CT head, this makes sure that there are no mask pixels around the outside of the brain mask, e.g.
        # in the corners.
        mask, _ = compute_contiguous_mask(mask=mask)

        # Fill mask holes
        mask = fill_mask_holes(mask=mask)

        # Visualise mask
        if self.debug:
            matshow3d(title=filename_or_obj, volume=mask, every_n=self.every_n, show=True, cmap="gray")

        # De transform masks
        mask = de_transform_mask(image, mask)

        # Add channel as SaveImage / NiftiSaver requires it
        mask = AddChannel()(mask)

        # Save mask
        output_dir = os.path.dirname(mask_path)
        meta["filename_or_obj"] = filename_or_obj
        SaveImage(output_dir=output_dir, output_postfix="", separate_folder=False)(mask, meta)

        return True
