#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/21/19
# Description: Models
# ========================================================

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import efficientnet_builder
import efficientnet_model


def get_inception_resnet_v2(training: bool=True, model_ckpt: str=None):
    """
    Build pre-trained keras Inception_Resnet_V2
    Returns: keras.Model

    """
    inputs = tf.keras.layers.Input(shape=(299, 299, 3), dtype=np.float32)
    inception_body = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights="imagenet",
                                                                                 input_tensor=inputs, pooling="avg")
    pred = tf.keras.layers.Dropout(0.4)(inception_body.output)
    pred = tf.keras.layers.Dense(5, activation="softmax", name="scores")(pred)

    return tf.keras.Model(inputs, pred)


def get_resnet_50(training: bool=True, model_ckpt: str=None):
    """
    Build pre-trained keras resnet_50
    Returns: keras.Model

    """
    inputs = tf.keras.layers.Input(shape=(256, 256, 3), dtype=np.float32)
    resnet_body = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet",
                                                          input_tensor=inputs, pooling="avg")

    pred = tf.keras.layers.Dropout(0.4)(resnet_body.output)
    pred = tf.keras.layers.Dense(5, activation="softmax", name="scores")(pred)

    return tf.keras.Model(inputs, pred)


def _get_efficientnet(images_tensor, model_name: str, training=True, model_ckpt: str=None):
    """
    Get efficientnet
    Args:
        images_tensor: input tensor
        model_name: name of model
        training: bool if it is for training
        model_ckpt: checkpoint file of model

    Returns:
        efficientnet feature extractor

    """
    with tf.variable_scope("Preprocess"):
        images_tensor = tf.cast(images_tensor, dtype=tf.float32)
        images_tensor -= tf.constant(efficientnet_builder.MEAN_RGB, shape=[1, 1, 3], dtype=images_tensor.dtype)
        images_tensor /= tf.constant(efficientnet_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=images_tensor.dtype)

    features, _ = efficientnet_builder.build_model_base(images_tensor, model_name=model_name, training=training)

    if model_ckpt:
        saver = tf.train.Saver()
        sess = K.get_session()
        saver.restore(sess, model_ckpt)

    return features


def get_efficientnet(model_name, training: bool=True, model_ckpt: str=None, regression=False):
    """ Build efficientnet_b0 and load pre-trained weights """
    model_param = efficientnet_builder.efficientnet_params(model_name)
    _, global_params = efficientnet_builder.get_model_params(model_name, {})
    image_size = model_param[2]

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3), dtype=tf.uint8, name="image_tensor")
    features = _get_efficientnet(inputs, model_name=model_name, training=training, model_ckpt=model_ckpt)

    with tf.variable_scope("head"):
        features = tf.keras.layers.Conv2D(filters=efficientnet_model.round_filters(1280, global_params),
                                          kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False)(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.ReLU()(features)
        features = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(features)

    if training:
        features = tf.keras.layers.Dropout(model_param[3])(features)

    if regression:
        logits = tf.keras.layers.Dense(1, names="scores")(features)
    else:
        logits = tf.keras.layers.Dense(5, activation="softmax", name="scores")(features)

    return tf.keras.Model(inputs, logits)
