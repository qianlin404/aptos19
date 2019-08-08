#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/7/19
# Description: 
# ========================================================

import imgaug.augmenters as iaa
import numpy as np

from PIL import Image, ImageOps
from typing import Callable, List


def cutout(images: np.ndarray, pad_size=16, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
    Returns:
    An image Tensor that is of type uint8.
    """

    if images.ndim == 4:
        images_aug = np.empty(images.shape, dtype=images.dtype)
        for i, image in enumerate(images):
            image_height = image.shape[0]
            image_width = image.shape[1]

            # Sample the center location in the image where the zero mask will be applied.
            cutout_center_height = np.random.randint(0, image_height, dtype=np.int32)

            cutout_center_width = np.random.randint(0, image_width, dtype=np.int32)

            lower_pad = np.max([0, cutout_center_height - pad_size])
            upper_pad = np.max([0, image_height - cutout_center_height - pad_size])
            left_pad = np.max([0, cutout_center_width - pad_size])
            right_pad = np.max([0, image_width - cutout_center_width - pad_size])

            cutout_shape = [image_height - (lower_pad + upper_pad),
                          image_width - (left_pad + right_pad)]
            padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
            mask = np.pad(
              np.zeros(cutout_shape, dtype=image.dtype),
              padding_dims, constant_values=1)
            mask = np.expand_dims(mask, -1)
            mask = np.tile(mask, [1, 1, 3])
            image = np.where(
              np.equal(mask, 0),
              np.ones_like(image, dtype=image.dtype) * replace,
              image)
            images_aug[i] = image
    elif images.ndim == 3:
        image = images
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = np.random.randint(0, image_height, dtype=np.int32)

        cutout_center_width = np.random.randint(0, image_width, dtype=np.int32)

        lower_pad = np.max([0, cutout_center_height - pad_size])
        upper_pad = np.max([0, image_height - cutout_center_height - pad_size])
        left_pad = np.max([0, cutout_center_width - pad_size])
        right_pad = np.max([0, image_width - cutout_center_width - pad_size])

        cutout_shape = [image_height - (lower_pad + upper_pad),
                        image_width - (left_pad + right_pad)]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = np.pad(
            np.zeros(cutout_shape, dtype=image.dtype),
            padding_dims, constant_values=1)
        mask = np.expand_dims(mask, -1)
        mask = np.tile(mask, [1, 1, 3])
        image = np.where(
            np.equal(mask, 0),
            np.ones_like(image, dtype=image.dtype) * replace,
            image)
        images_aug = image
    else:
        raise RuntimeError("Unexpecte shape {}".format(images.shape))

    return images_aug


def cutout_wrap(images,random_state, parents, hooks):
    """ Cutout warpper """
    return cutout(images)


def equalize(images: np.ndarray):
    """
    Perform equalize operation to images
    Args:
        images: images tensor ndarray

    Returns:
        aug_images: augmented images

    """
    # image batches
    if images.ndim == 4:
        aug_images = np.empty(shape=images.shape, dtype=images.dtype)
        for i, image in enumerate(images):
            image_obj = Image.fromarray(image)
            image_obj = ImageOps.equalize(image_obj)
            aug_images[i] = np.array(image_obj, dtype=images.dtype)

    # single image
    elif images.ndim == 3:
        aug_images = Image.fromarray(images)
        aug_images = ImageOps.equalize(aug_images)
        aug_images = np.array(aug_images, dtype=images.dtype)

    else:
        raise RuntimeError("Unknow image shape {}".format(images.shape))

    return aug_images


def solarize(images: np.ndarray, threshold: int):
    """
    solarize image with threshold
    Args:
        images: image tensor ndarray
        threshold: int

    Returns:
        aug_images: augmented images
    """
    # image batches
    if images.ndim == 4:
        aug_images = np.empty(shape=images.shape, dtype=images.dtype)
        for i, image in enumerate(images):
            image_obj = Image.fromarray(image)
            image_obj = ImageOps.solarize(image_obj, threshold)
            aug_images[i] = np.array(image_obj, dtype=images.dtype)

    elif images.ndim == 3:
        aug_images = Image.fromarray(images)
        aug_images = np.array(ImageOps.solarize(aug_images, threshold), dtype=images.dtype)

    else:
        raise RuntimeError("Unknow image shape {}".format(images.shape))
    return aug_images


def posterize(images: np.ndarray, bit: int):
    """
    solarize image with threshold
    Args:
        images: image tensor ndarray
        bit: int

    Returns:
        aug_images: augmented images
    """
    # image batches
    if images.ndim == 4:
        aug_images = np.empty(shape=images.shape, dtype=images.dtype)
        for i, image in enumerate(images):
            image_obj = Image.fromarray(image)
            image_obj = ImageOps.posterize(image_obj, bit)
            aug_images[i] = np.array(image_obj, dtype=images.dtype)

    elif images.ndim == 3:
        aug_images = Image.fromarray(images)
        aug_images = ImageOps.posterize(aug_images, bit)
        aug_images = np.array(aug_images, dtype=images.dtype)

    else:
        raise RuntimeError("Unknow image shape {}".format(images.shape))
    return aug_images


def color(images: np.ndarray):
    """
    To gray scale
    Args:
        images: ndarray image tensor

    Returns:
        aug_images
    """
    # image batches
    if images.ndim == 4:
        aug_images = np.empty(shape=images.shape, dtype=images.dtype)
        for i, image in enumerate(images):
            image_obj = Image.fromarray(image).convert('L').convert("RGB")
            aug_images[i] = np.array(image_obj, dtype=images.dtype)

    elif images.ndim == 3:
        aug_images = Image.fromarray(images).convert('L').convert("RGB")
        aug_images = np.array(aug_images, dtype=images.dtype)

    else:
        raise RuntimeError("Unknow image shape {}".format(images.shape))
    return aug_images


class AugPolicy(object):
    """ imgaug image augmentation policy """
    def __init__(self, operations: List, num_subpolicy: int):
        """
        Initializer
        Args:
            operations: list of available operations
            num_subpolicy: number sub-policy
        """
        self.operations = operations
        self.num_subpolicy = num_subpolicy

    def generate_policy(self):
        """
        generate policy
        Returns:
            policy: List
            policy_config: Dict
        """
        policy = []
        policy_config = []

        for i in range(self.num_subpolicy):
            index = np.random.randint(0, len(self.operations), 2)
            subpolicy = [self.operations[int(i)] for i in index]
            policy.append(subpolicy)
            policy_config.append([p.name for p in subpolicy])

        return policy, dict(policy=policy_config)


def get_operation_pool():
    """ Get list of operations """

    def equalize_wrap(images,random_state, parents, hooks):
        return equalize(images)

    def color_wrap(images,random_state, parents, hooks):
        return color(images)

    def solarize_60_wrap(images,random_state, parents, hooks):
        return solarize(images, threshold=60)

    def solarize_80_wrap(images,random_state, parents, hooks):
        return solarize(images, threshold=80)

    def posterize_wrap_3(images,random_state, parents, hooks):
        return posterize(images, bit=3)

    def posterize_wrap_6(images,random_state, parents, hooks, ):
        return posterize(images, bit=6)

    return [
        iaa.Fliplr(.5, name="fliplr_0.5"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=equalize_wrap), name="equalize_0.3"),
        iaa.Sometimes(.7), iaa.Lambda(func_images=equalize_wrap, name="euqalize_0.7"),
        iaa.Sometimes(.3, iaa.Affine(rotate=(-30, 30)), name="rotate_30_0.3"),
        iaa.Sometimes(.7, iaa.Affine(rotate=(-30, 30)), name="rotate_30_0.7"),
        iaa.Sometimes(.3, iaa.Affine(shear=(-16, 16)), name="shear_16_0.3"),
        iaa.Sometimes(.7, iaa.Affine(shear=(-16, 16)), name="shear_16_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=color_wrap), name="color_0.3"),
        iaa.Sometimes(.7, iaa.Lambda(func_images=color_wrap), name="color_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=solarize_60_wrap), name="solarize_60_0.3"),
        iaa.Sometimes(.7, iaa.Lambda(func_images=solarize_60_wrap), name="solarize_60_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=solarize_80_wrap), name="solarize_80_0.3"),
        iaa.Sometimes(.7, iaa.Lambda(func_images=solarize_80_wrap), name="solarize_80_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=posterize_wrap_3), name="posterize_3_0.3"),
        iaa.Sometimes(.7, iaa.Lambda(func_images=posterize_wrap_3), name="posterize_3_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=posterize_wrap_6), name="posterize_6_0.3"),
        iaa.Sometimes(.7, iaa.Lambda(func_images=posterize_wrap_6), name="posterize_6_0.7"),
        iaa.Sometimes(.3, iaa.Lambda(func_images=cutout_wrap, name="cutout_0.3")),
        iaa.Sometimes(.7, iaa.Lambda(func_images=cutout_wrap, name="cutout_0.7"))
    ]


def generate_iaa_sequence(policy: List):
    """ Randomly generate image augment sequence """
    head_aug = [iaa.Sometimes(.5, iaa.OneOf([iaa.Add((-10, 10), per_channel=0.5),
                                            iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)]))]

    body_aug = policy[np.random.randint(0, len(policy))]
    tail_aug = [iaa.Lambda(func_images=cutout_wrap, name="cutout")]
    
    seq = iaa.Sequential(head_aug+body_aug+tail_aug)

    return seq
