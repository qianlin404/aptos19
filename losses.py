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
        weights = tf.log(pred_distance+tf.exp(1.0), dtype=tf.float32)

        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * weights
