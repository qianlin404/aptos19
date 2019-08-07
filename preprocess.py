#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/21/19
# Description: Preprocess functions
# ========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from PIL import Image
from typing import List, Tuple, Callable
from functools import partial


def load_default(filenames: List[str], size=(299, 299)) -> np.ndarray:
    """
    Load image into tensor given filenames
    Args:
        filenames: List of filename strings
        size: Tuple in (height, width) format

    Returns:
        image_tensor
    """

    res = [cv2.resize(_load_image(f), size) for f in filenames]
    return res


def _load_image(filename: str) -> np.ndarray:
    """
    Load single image and resize
    Args:
        filename: filename of the image
        size: Tuple in (height, width) format

    Returns:
        image_tensor
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_ben_color(path: str, image_size: Tuple, sigmaX: int=10):
    """
    Perform ben's preprocess method
    Args:
        path: path to image file
        image_size: size of the output image
        sigmaX: sigmaX for x weight

    Returns:

    """
    image = _load_image(path)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, image_size)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def crop_image_from_gray(img: np.ndarray, tol=7):
    """
    Crop the image from gray
    Args:
        img: ndarray, in (H, W, C) format
        tol: int

    Returns:

    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def mean_std_normalization(images: List[np.ndarray], mean_value: Tuple=(105.53, 56.36, 18.79),
                           std_value: Tuple=(60.93, 33.67, 12.66)) -> np.ndarray:
    """
    Perform mean std normalization
    Args:
        images: list of ndarray in [H, W, C] format
        mean_value: 3-Tuple of mean values for (R, G, B) channels
        std_value: 3-Tuple of std values for (R, G, B) channels

    Returns:
        normalized_tensor: normalized tensor
    """
    tensor = np.concatenate([np.expand_dims(img, axis=0) for img in images], axis=0)
    return (tensor - mean_value) / std_value


def crop_and_ben_normalized(filenames: List[str], image_size: Tuple):
    """
    Crop and do ben's preprocess method
    Args:
        filenames: filename of images
        image_size: image size in (H, W) format

    Returns:
        preprocessed_image:
    """
    preprocessed_image = [load_ben_color(filename, image_size) for filename in filenames]
    preprocessed_image = [crop_image_from_gray(image) for image in preprocessed_image]

    return preprocessed_image


class ImageGenerator(tf.keras.utils.Sequence):
    """ Generate batched images """
    def __init__(self, df: pd.DataFrame , batch_size: int, image_size: Tuple, load_fn: Callable, augment_fn: Callable,
                 preprocess_fn: Callable, is_test: bool=False, is_augment: bool=False, seed=404):
        self._image_data = df
        self._batch_size = batch_size
        self._image_size = image_size
        self._preprocess_fn = preprocess_fn
        self._seed = seed
        self._is_test = is_test
        self._is_augment = is_augment
        self._load_fn = load_fn
        self._augment_fn = augment_fn

        np.random.seed(self._seed)
        indexes = np.array([i for i in range(self.__len__() * self._batch_size)])
        self._batch_index = np.random.permutation(indexes).reshape((self.__len__(), self._batch_size))

    def __len__(self):
        return self._image_data.shape[0] // self._batch_size

    def __getitem__(self, item):
        """ Return a generator that generate (images, label) tuple """
        i = self._batch_index[item]
        images = self._load_fn(self._image_data["path"].iloc[i], size=self._image_size)

        if self._is_test:
            labels = None
        else:
            labels = self._image_data.iloc[i]["diagnosis"].values

        if self._is_augment:
            images, labels = self._augment_fn(images, labels)

        images = self._preprocess_fn(images)

        return images, labels


