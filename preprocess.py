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

from PIL import Image
from typing import List, Tuple, Callable
from multiprocessing import Pool
from functools import partial


def load_iamges(filenames: List[str], size=(299, 299)) -> np.ndarray:
    """
    Load image into tensor given filenames
    Args:
        filenames: List of filename strings
        size: Tuple in (height, width) format

    Returns:
        image_tensor
    """
    pool = Pool(10)
    load_func = partial(_load_images, size=size)

    res = pool.map(load_func, filenames)
    pool.close()

    return np.concatenate(res)


def _load_images(filename: str, size) -> np.ndarray:
    """
    Load single image and resize
    Args:
        filename: filename of the image
        size: Tuple in (height, width) format

    Returns:
        image_tensor
    """
    img = Image.open(filename).resize(size)
    return np.expand_dims(np.array(img), axis=0)


def mean_std_normalization(tensor: np.ndarray, mean_value: Tuple=(105.53, 56.36, 18.79),
                           std_value: Tuple=(60.93, 33.67, 12.66)) -> np.ndarray:
    """
    Perform mean std normalization
    Args:
        tensor: batched image array in [N, H, W, C] format
        mean_value: 3-Tuple of mean values for (R, G, B) channels
        std_value: 3-Tuple of std values for (R, G, B) channels

    Returns:
        normalized_tensor: normalized tensor
    """
    return (tensor - mean_value) / std_value


class ImageGenerator(tf.keras.utils.Sequence):
    """ Generate batched images """
    def __init__(self, df: pd.DataFrame , batch_size: int, image_size: Tuple, preprocess_fn: Callable, seed=404):
        self._image_data = df
        self._batch_size = batch_size
        self._image_size = image_size
        self._preprocess_fn = preprocess_fn
        self._seed = seed

        np.random.seed(self._seed)
        indexes = np.array([i for i in range(self.__len__() * self._batch_size)])
        self._batch_index = np.random.permutation(indexes).reshape((self.__len__(), self._batch_size))

    def __len__(self):
        return self._image_data.shape[0] // self._batch_size

    def __getitem__(self, item):
        """ Return a generator that generate (images, label) tuple """
        i = self._batch_index[item]
        labels = self._image_data.iloc[i]["diagnosis"].values
        images = load_iamges(self._image_data["path"].iloc[i], size=self._image_size)
        images = self._preprocess_fn(images)

        return (images, labels)