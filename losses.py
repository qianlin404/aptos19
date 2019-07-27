#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/27/19
# Description: 
# ========================================================

import tensorflow as tf


def quadratic_kappa(y_true, y_pred):
    """ Quadratic kappa """
    pred = tf.math.argmax(y_pred, axis=1)

    confusion = tf.confusion_matrix(y_true, pred, dtype=tf.float32)
    n_classes = tf.shape(confusion)[0]
    sum0 = tf.reduce_sum(confusion, axis=0)
    sum1 = tf.reduce_sum(confusion, axis=1)

    expected = tf.reshape(sum0, [n_classes, 1]) * tf.reshape(sum1, [1, n_classes]) / tf.reduce_sum(sum0)

    w_mat = tf.zeros([n_classes, n_classes], dtype=tf.float32)
    w_mat += tf.range(tf.cast(n_classes, dtype=tf.float32), dtype=tf.float32)
    w_mat = (w_mat - tf.transpose(w_mat)) ** 2

    k = tf.reduce_sum(w_mat * confusion) / tf.reduce_sum(w_mat * expected)

    return 1 - k