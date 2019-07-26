#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/25/19
# Description: 
# ========================================================

import tensorflow as tf


def regularizer_l2(model, lamb):
    """ L2 regularizer """
    for layer in model.layers:
        layer.kernel_regularizer = tf.keras.regularizers.l2(lamb)
