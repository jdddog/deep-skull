"""
Derived from here: https://github.com/aqqush/CT_BET
Using model weights provided by the CT_BET project as well
"""

import os
from os.path import expanduser

import numpy as np
from tensorflow.keras.layers import (
    Activation,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file

WEIGHTS_NAME = "unet_CT_SS_20171114_170726.h5"
WEIGHTS_URL = (
    "https://onedrive.live.com/download?cid=6917C8254765425B&resid=6917C8254765425B%21152&authkey=AFhai1pqNU1ndTc"
)
WEIGHTS_HASH = "f45adda070688efacb2d0a5d7e72c8ba"
MODEL_KEY = "ct_bet"


def get_weights_path() -> str:
    cache_dir = os.path.join(expanduser("~"), ".keras")
    return get_file(WEIGHTS_NAME, WEIGHTS_URL, cache_subdir="models", md5_hash=WEIGHTS_HASH, cache_dir=cache_dir)


class BrainExtractorModel:
    def __init__(self, input_shape=(512, 512, 1), num_classes=2, weights="ct_bet", batch_size: int = 8):
        self.input_shape = input_shape
        self.input_row = input_shape[0]
        self.input_col = input_shape[1]
        self.num_classes = num_classes
        self.weights = weights
        self.batch_size = batch_size

        if (weights not in [MODEL_KEY, None]) and not os.path.exists(weights):
            raise ValueError("The `weights` parameter must be either None, a valid path or `ct_bet`.")

        self.model = self.__get_unet()

    def predict(self, slices):
        sc = 2  # Not sure what this does
        num_slices = len(slices)
        slices = slices.reshape(
            num_slices, slices.shape[1], slices.shape[2], 1
        )  # Have to reshape for the network for some reason?
        brain_masks = self.model.predict(slices, batch_size=self.batch_size, verbose=1)
        brain_masks = brain_masks.reshape((num_slices, self.input_row, self.input_col, self.num_classes))[
            :, :, :, sc - 1 : sc
        ]
        brain_masks = (brain_masks > 0.5).astype(np.float).reshape(num_slices, self.input_row, self.input_col)
        return brain_masks

    def __get_unet(self):
        # Input for network
        inputs = Input(shape=self.input_shape)

        # Drop network
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(inputs)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(pool1)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(pool2)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(pool3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(pool4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Up network
        up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(drop5)
        merge6 = concatenate([up6, drop4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(merge6)
        conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv6)

        up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv6)
        merge7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(merge7)
        conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv7)

        up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(conv7)
        merge8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(merge8)
        conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv8)

        up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(conv8)
        merge9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(merge9)
        conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="glorot_uniform")(conv9)
        # The last conv9 was removed? Why?
        conv10 = Conv2D(self.num_classes, (1, 1), activation="relu")(conv9)

        # Build model
        base_model = Model(inputs=[inputs], outputs=[conv10])
        act1 = Activation("softmax")(base_model.output)
        new_output = Reshape((self.input_row * self.input_col, 1, self.num_classes))(act1)
        top_model = Model(base_model.input, new_output)
        top_model.compile(
            optimizer=Adam(lr=1e-5, decay=1e-6),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            sample_weight_mode="temporal",
        )

        if self.weights == MODEL_KEY:
            weights_path = get_weights_path()
            top_model.load_weights(weights_path)
        elif self.weights is not None:
            top_model.load_weights(self.weights)

        return top_model
