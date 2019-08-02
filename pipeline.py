#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 7/25/19
# Description: Pipeline class
# ========================================================

import os
import time
import numpy as np
import pandas as pd
import json
import preprocess
import tensorflow as tf

from pathlib import Path
from functools import partial
from typing import Callable, Tuple, Dict, List


def get_image_paths(id_code, image_dir):
    """
    Get image path by concatenating image directory with image code id and suffix
    Args:
        image_dir: image directory
        id_code: image code ID

    Returns:
        image_path: str

    """
    suffix = ".png"
    return os.path.join(image_dir, id_code + suffix)


class KerasPipeline(object):
    """ Pileline for training and evaluating model """
    def __init__(self,
                 training_filename: str,
                 validation_filename: str,
                 image_dir: str,
                 preprocess_fn: Callable,
                 image_size: Tuple,
                 batch_size: int,
                 model_generating_fn: Callable,
                 optimizer: Callable,
                 optimizer_params: Dict,
                 num_epochs: int,
                 loss: Callable,
                 regularizer: Callable,
                 regularizer_params: Dict,
                 eval_metrics: List[Callable],
                 cv_fn: Callable,
                 name: str):
        """
        Initializer
        Args:
            training_filename: training set file path
            validation_filename: validation set file path
            image_dir: image directory
            preprocess_fn: preprocess function
            image_size: image size in (H, W) format
            batch_size: batch size for training
            model_generating_fn: function to generate the model
            optimizer: optimizer to train the model
            optimizer_params: parameters of optimizer
            num_epochs: how many epochs to train
            loss: loss function
            regularizer: regularizer function
            regularizer_params: regularizer parameters
            eval_metrics: evaluation metrics
            cv_fn: Cross validation function
            name: name of this run
        """
        self.training_filename = training_filename
        self.validation_filename = validation_filename
        self.image_dir = image_dir
        self.preprocess_fn = preprocess_fn
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_generating_fn = model_generating_fn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.num_epochs = num_epochs
        self.loss = loss
        self.regularizer = regularizer
        self.regularizer_params = regularizer_params
        self.eval_metrics = eval_metrics
        self.cv_fn = cv_fn
        self.name = name

        # Placeholders
        self.training_set = None
        self.validation_set = None
        self.training_samples = None
        self.validation_samples = None
        self.model = None
        self.train_generator = None
        self.val_generator = None

        self.eval = {}
        self.created_time = int(time.time())

    def write_config(self):
        input_config = dict(training_filename=self.training_filename,
                            validation_filename=self.validation_filename,
                            image_dir=self.image_dir,
                            training_samples=self.training_samples,
                            validation_samples=self.validation_samples)
        preprocess_config = dict(preprocess_func=self.preprocess_fn.__name__,
                                 image_size=self.image_size)
        optimizer_config = dict(name=self.optimizer.__name__, params=self.optimizer_params)
        regularizer_config = dict(name=self.regularizer.__name__, params=self.regularizer_params)
        train_config = dict(optimizer=optimizer_config, num_epochs=self.num_epochs, loss=self.loss.__name__,
                            regularizer=regularizer_config)
        evaluation = self.eval

        config = dict(input_data=input_config, preprocess=preprocess_config, train_config=train_config,
                      evaluation=evaluation)

        save_dir = self._get_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, "config.json")

        print("[INFO] Saving config to {}".format(save_filename))
        with open(save_filename, 'w') as f:
            f.write(json.dumps(config))

    def _get_save_dir(self):
        folder_name = ""
        for k, v in self.eval.items():
            folder_name += "{k}={v:.2f}_".format(k=k, v=v)
        folder_name += str(self.created_time)

        return str(Path("models/") / folder_name)

    def _read_input(self):
        """ Run input """
        self.training_set = pd.read_csv(self.training_filename)
        self.validation_set = pd.read_csv(self.validation_filename)
        print("{t:<20}: {training_filename}".format(t="Training set", training_filename=self.training_filename))
        print("{t:<20}: {training_filename}".format(t="valiation set", training_filename=self.validation_filename))

        get_path_fn = partial(get_image_paths, image_dir=self.image_dir)
        self.training_set["path"] = self.training_set["id_code"].apply(get_path_fn)
        self.validation_set["path"] = self.validation_set["id_code"].apply(get_path_fn)

        self.training_samples = self.training_set.shape[0]
        self.validation_samples = self.validation_set.shape[0]
        print("{t:<20}: {size}".format(t="Training set size", size=self.training_samples))
        print("{t:<20}: {size}".format(t="Validation set size", size=self.validation_samples))

    def _get_input_generator(self):
        """ Return input generator for trainning validation data """
        train = preprocess.ImageGenerator(self.training_set, image_size=self.image_size, batch_size=self.batch_size,
                                          preprocess_fn=self.preprocess_fn)
        val = preprocess.ImageGenerator(self.validation_set, image_size=self.image_size, batch_size=1,
                                        preprocess_fn=self.preprocess_fn)
        print("{t:<20}: {batch_size}".format(t="Batch size", batch_size=self.batch_size))
        print("{t:<20}: {image_size}".format(t="Image size", image_size=self.image_size))

        return train, val

    def _build_model(self):
        """ Build and compile model """
        print("{t:<20}: {model}".format(t="Model", model=self.model_generating_fn.__name__))
        model = self.model_generating_fn()

        print("{t:<20}: {reg}".format(t="Regularizer", reg=self.regularizer.__name__))
        print(json.dumps(self.regularizer_params))
        self.regularizer(model, **self.regularizer_params)

        print("{t:<20}: {opt}".format(t="Optimizer", opt=self.optimizer.__name__))
        print(json.dumps(self.optimizer_params))
        optimizer = self.optimizer(**self.optimizer_params)

        print("{t:<20}: {loss}".format(t="Loss Function", loss=self.loss.__name__))
        model.compile(optimizer, loss=self.loss, metrics=self.eval_metrics)

        return model

    def _cv(self):
        """ Cross validation """
        y_true = []
        y_pred = []

        for i in range(len(self.val_generator)):
            image, label = self.val_generator[i]
            pred = self.model.predict(image)
            pred = np.argmax(pred, axis=1).ravel()
            y_true.append(label)
            y_pred.append(pred)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        score = self.cv_fn(y_true, y_pred)

        print("{metric_name}: {score:.2f}".format(metric_name=self.cv_fn.__name__, score=score))
        return score

    def train(self):
        self._read_input()
        self.train_generator, self.val_generator = self._get_input_generator()
        self.model = self._build_model()

        logdir = "tensorboard/" + self.name + "_" + str(self.created_time)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, update_freq="batch")

        self.model.fit_generator(self.train_generator, steps_per_epoch=len(self.train_generator),
                                 validation_data=self.val_generator, validation_steps=len(self.val_generator),
                                 epochs=self.num_epochs, callbacks=[tb_callback])

        metric_value = self._cv()

        self.eval[self.cv_fn.__name__] = metric_value

        self.write_config()

    def save(self):
        tf.contrib.saved_model.save_keras_model(self.model, self._get_save_dir())

