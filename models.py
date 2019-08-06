#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/21/19
# Description: Models
# ========================================================

import tensorflow as tf
import numpy as np


def get_inception_resnet_v2():
    """
    Build pre-trained keras Inception_Resnet_V2
    Returns: keras.Model

    """
    inputs = tf.keras.layers.Input(shape=(299, 299, 3), dtype=np.float32)
    inception_body = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights="imagenet",
                                                                                 input_tensor=inputs, pooling="avg")
    pred_dense = tf.keras.layers.Dense(5, activation="softmax", name="scores")
    pred = pred_dense(inception_body.output)

    return tf.keras.Model(inputs, pred)


def get_resnet_50():
    """
    Build pre-trained keras resnet_50
    Returns: keras.Model

    """
    inputs = tf.keras.layers.Input(shape=(256, 256, 3), dtype=np.float32)
    resnet_body = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet",
                                                          input_tensor=inputs, pooling="avg")
    pred_dense = tf.keras.layers.Dense(5, activation="softmax", name="scores")
    pred = pred_dense(resnet_body.output)

    return tf.keras.Model(inputs, pred)

