#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/27/19
# Description: 
# ========================================================

import tensorflow as tf


def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    """
    weighted version of sparse categorical crossentropy loss
    Args:
        y_true: ground true labels, in shape [batch_size]
        y_pred: prediction logits, in shape [batch_size, num_classes]

    Returns:
        loss: tensor

    """
    with tf.variable_scope("weighted_categorical_crossentropy"):
        pred = tf.argmax(y_pred, axis=1, name="prediction")
        labels = tf.cast(y_true, dtype=tf.int64)

        pred_distance = tf.abs(labels-pred, name="distance")
        weights = tf.log(tf.cast(pred_distance, tf.float32)+tf.exp(1.0))

        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights


def clip_meam_square_error(y_true, y_pred):
    """
    Clip the logit and compute MSE
    Args:
        y_true: ground true labels, in shape [batch_size]
        y_pred: logit, in shape [batch_size, 1]

    Returns:
        loss: tensor
    """
    with tf.variable_scope("clip_mean_squared_error"):
        pred = tf.clip_by_value(y_pred, clip_value_min=-0.5, clip_value_max=4.5)

        return tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=pred)