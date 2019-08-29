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
import sklearn

from scipy import optimize
from pathlib import Path
from functools import partial
from typing import Callable, Tuple, Dict, List


def QWK(y_true, y_pred):
    """ Quadratic weighted kappa """
    pred = tf.cast(y_pred, tf.int32)
    pred = tf.clip_by_value(tf.round(pred), 0, 4)
    pred = tf.squeeze(pred)

    confusion = tf.math.confusion_matrix(y_true, pred, num_classes=5, dtype=tf.int32)
    n_classes = 5

    sum0 = tf.reduce_sum(confusion, axis=0)
    sum1 = tf.reduce_sum(confusion, axis=1)
    expected = tf.einsum('i,j->ij', sum0, sum1) / tf.reduce_sum(sum0)

    w_mat = tf.zeros([n_classes, n_classes], dtype=tf.int32)
    w_mat += tf.range(n_classes)
    w_mat = (w_mat - tf.transpose(w_mat)) ** 2

    k = tf.cast(tf.reduce_sum(w_mat * confusion), tf.double) / tf.reduce_sum(tf.cast(w_mat, tf.double) * expected)
    return 1 - k


class EarlyStopKappaCallback(tf.keras.callbacks.Callback):
    """ Early stop based on kappa score """
    def __init__(self, eval_func, save_path, patience):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.eval_func = eval_func
        self.save_path = save_path
        self.highest_score = -1
        self.cnt = 0
        self.patient = patience

    def on_epoch_end(self, epoch, logs=None):
        cur_score = self.eval_func()
        print("\n[INFO] Quadratic weighted kappa: %.4f" % cur_score)

        if cur_score > self.highest_score:
            print("\n[INFO] Hit a higher score {score}, saving model...".format(score=cur_score))
            self.model.save_weights(self.save_path)
            self.highest_score = cur_score
            self.cnt = 0
        else:
            self.cnt += 1

        if self.cnt > self.patient:
            print("\n[INFO] score stop increasing, early stopping...")
            self.model.stop_training = True

            print("[INFO] restoring best model, quadratic weighted kappa: %.4f" % self.highest_score)
            self.model.load_weights(self.save_path, by_name=True)


class Postprocessor(object):
    """ convert model output to predicted labels """
    def get_predition(self, output_tensor: np.ndarray):
        """
        Model output to prediction labels
        Args:
            output_tensor: raw output from model, can be probability or logits

        Returns:
            predicted_labels: labels, int, in shape [batch_size,]

        """
        raise NotImplementedError


class ClassificationPostprocessor(Postprocessor):
    """ Postprocessor for classification """
    def get_predition(self, output_tensor: np.ndarray):
        """ Convert probability to predicted labels"""
        return np.argmax(output_tensor, axis=1).ravel()


class RegressionPostprocessor(Postprocessor):
    """ Postprocessor for regression """
    def __init__(self, threshold=None):
        if threshold:
            self.threshold = np.array(threshold, dtype=np.float32)
        else:
            self.threshold = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        self.optimizer = None

    @staticmethod
    def _get_one_prediction(threshold, value):
        """ get prediction using threshold """
        if value < threshold[0]:
            return 0
        elif threshold[0] <= value < threshold[1]:
            return 1
        elif threshold[1] <= value < threshold[2]:
            return 2
        elif threshold[2] <= value < threshold[3]:
            return 3
        else:
            return 4

    def get_predition(self, output_tensor: np.ndarray):
        """ Convert regression value to predicted labels """
        predicted_labels = np.zeros(shape=(output_tensor.shape[0]), dtype=np.int32)

        for i, pred in enumerate(output_tensor):
            predicted_labels[i] = self._get_one_prediction(self.threshold, pred)

        return predicted_labels

    def fit(self, predicted_logits: np.ndarray, labels: np.ndarray):
        """
        Find threshold the optimize quadratic weighted kappa scores
        Args:
            predicted_logits: logits from model
            labels: ground true labels

        Returns:
            None, update self.threshold
        """
        origin_prediction = self.get_predition(predicted_logits)
        origin_kappa = sklearn.metrics.cohen_kappa_score(labels, origin_prediction, weights="quadratic")
        print("Origin QWK is: %.4f" % origin_kappa)

        def _kappa_loss(threshold):
            """ function use for optimization """
            predicted_labels = np.zeros(shape=(predicted_logits.shape[0]), dtype=np.int32)
            for i, pred in enumerate(predicted_logits):
                predicted_labels[i] = self._get_one_prediction(threshold, pred)

            return -sklearn.metrics.cohen_kappa_score(labels, predicted_labels, weights="quadratic")

        self.optimizer = optimize.minimize(_kappa_loss, self.threshold, method='nelder-mead')
        self.threshold = self.optimizer['x']

        optimized_prediction = self.get_predition(predicted_logits)
        optimized_kappa = sklearn.metrics.cohen_kappa_score(labels, optimized_prediction, weights="quadratic")
        print("Optimized QWK is: %.4f" % optimized_kappa)


def get_image_paths(id_code, image_dir, suffix=".png"):
    """
    Get image path by concatenating image directory with image code id and suffix
    Args:
        image_dir: image directory
        id_code: image code ID
        suffix: suffix of file

    Returns:
        image_path: str

    """
    return os.path.join(image_dir, id_code + suffix)


class KerasPipeline(object):
    """ Pileline for training and evaluating model """
    def __init__(self,
                 training_filename: str,
                 validation_filename: str,
                 train_image_dir: str,
                 load_fn: Callable,
                 augment_policy: List,
                 augment_config: Dict,
                 preprocess_fn: Callable,
                 postprocessor: Postprocessor,
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
                 name: str,
                 record_name: str,
                 model_weights_filename=None,
                 val_image_dir=None,
                 model_ckpt: str=None,
                 train_image_suffix: str=".png",
                 val_image_suffix: str=".png",
                 fine_tuning_layers: int=None,
                 multi_gpu=1):
        """
        Initializer
        Args:
            training_filename: training set file path
            validation_filename: validation set file path
            train_image_dir: image directory
            load_fn: function to load image
            augment_fn: function to perform data augmentation
            preprocess_fn: preprocess function
            postprocessor: Postprocessor object
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
            record_name: name of this running
            model_weights_filename: filename of model weights, default to none
            val_image_dir: validation set image directory
            model_ckpt: checkpoint prefix
            train_image_suffix: suffix of image files
            fine_tuning_layers: last number of fine tuning layers
            multi_gpu: the number of gpus
        """
        self.training_filename = training_filename
        self.validation_filename = validation_filename
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir if val_image_dir else train_image_dir
        self.load_fn = load_fn
        self.augment_policy = augment_policy
        self.augment_config = augment_config
        self.preprocess_fn = preprocess_fn
        self.postprocessor = postprocessor
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
        self.model_weights_filename = model_weights_filename
        self.model_ckpt = model_ckpt
        self.train_image_suffix = train_image_suffix
        self.val_image_suffix = val_image_suffix
        self.fine_tuning_layers = fine_tuning_layers
        self.multi_gpu = multi_gpu

        # Placeholders
        self.training_set = None
        self.validation_set = None
        self.training_samples = None
        self.validation_samples = None
        self.model = None
        self.train_generator = None
        self.val_generator = None

        self.eval = {}
        self.record_name = record_name
        os.makedirs(self._get_save_dir(), exist_ok=True)

        self.ckpt_path = os.path.join(self._get_save_dir(), "weights.best.h5")

    def write_config(self):
        input_config = dict(training_filename=self.training_filename,
                            validation_filename=self.validation_filename,
                            image_dir=self.train_image_dir,
                            training_samples=self.training_samples,
                            validation_samples=self.validation_samples)
        preprocess_config = dict(load_fn=self.load_fn.__name__,
                                 preprocess_func=self.preprocess_fn.__name__,
                                 image_size=self.image_size,
                                 augment_config=self.augment_config)
        optimizer_config = dict(name=self.optimizer.__name__, params=self.optimizer_params)
        regularizer_config = dict(name=self.regularizer.__name__, params=self.regularizer_params)
        train_config = dict(optimizer=optimizer_config, num_epochs=self.num_epochs, loss=self.loss.__name__,
                            regularizer=regularizer_config)
        evaluation = self.eval
        model_config = dict(name=self.name, checkpoint=self.model_ckpt)

        config = dict(input_data=input_config, preprocess=preprocess_config, train_config=train_config,
                      evaluation=evaluation, model=model_config)

        save_dir = self._get_save_dir()
        save_filename = os.path.join(save_dir, "config.json")

        print("[INFO] Saving config to {}".format(save_filename))
        with open(save_filename, 'w') as f:
            f.write(json.dumps(config))

    def _get_save_dir(self):
        folder_name = os.path.join("models", self.record_name)
        return folder_name

    def _get_callback(self):
        """ Get callbacks for the model """
        early_stop_kappa = EarlyStopKappaCallback(self._quadratic_weighted_kappa,
                                                  save_path=self.ckpt_path,
                                                  patience=5)

        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=3, mode="min",
                                                        verbose=True)

        logdir = "tensorboard/" + self.record_name
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, update_freq="epoch")

        return [early_stop_kappa, lr_decay, tensorboard]

    def _read_input(self):
        """ Run input """
        self.training_set = pd.read_csv(self.training_filename)
        self.validation_set = pd.read_csv(self.validation_filename)
        print("{t:<20}: {training_filename}".format(t="Training set", training_filename=self.training_filename))
        print("{t:<20}: {training_filename}".format(t="valiation set", training_filename=self.validation_filename))

        get_train_path_fn = partial(get_image_paths, image_dir=self.train_image_dir, suffix=self.train_image_suffix)
        get_val_path_fn = partial(get_image_paths, image_dir=self.val_image_dir, suffix=self.val_image_suffix)
        self.training_set["path"] = self.training_set["id_code"].apply(get_train_path_fn)
        self.validation_set["path"] = self.validation_set["id_code"].apply(get_val_path_fn)

        self.training_samples = self.training_set.shape[0]
        self.validation_samples = self.validation_set.shape[0]
        print("{t:<20}: {size}".format(t="Training set size", size=self.training_samples))
        print("{t:<20}: {size}".format(t="Validation set size", size=self.validation_samples))

    def _get_input_generator(self):
        """ Return input generator for trainning validation data """
        print("Loading trainning data...")
        train = preprocess.ImageGenerator(self.training_set, image_size=self.image_size, batch_size=self.batch_size,
                                          preprocess_fn=self.preprocess_fn, augment_policy=self.augment_policy,
                                          load_fn=self.load_fn, is_augment=True)
        train.show_sample()

        print("Loading validation data...")
        val = preprocess.ImageGenerator(self.validation_set, image_size=self.image_size, batch_size=self.batch_size,
                                        preprocess_fn=self.preprocess_fn, augment_policy=self.augment_policy,
                                        load_fn=self.load_fn, is_augment=False)
        val.show_sample()
        print("{t:<20}: {batch_size}".format(t="Batch size", batch_size=self.batch_size))
        print("{t:<20}: {image_size}".format(t="Image size", image_size=self.image_size))

        return train, val

    def _build_model(self):
        """ Build and compile model """
        print("{t:<20}: {model}".format(t="Model", model=self.name))
        # if self.multi_gpu > 1:
        #     with tf.device("/cpu:0"):
        #         model = self.model_generating_fn(training=True, model_ckpt=self.model_ckpt)
        # else:
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = self.model_generating_fn(training=True, model_ckpt=self.model_ckpt)

            if self.model_weights_filename:
                print("{t:<20}: {filename}".format(t="Model weights", filename=self.model_weights_filename))
                model.load_weights(self.model_weights_filename, by_name=True)

            if self.fine_tuning_layers:
                for layer in model.layers[:-self.fine_tuning_layers]:
                    layer.trainable = False
                    print("[INFO] Layer {name} is now non-trainable".format(name=layer.name))

            print("{t:<20}: {reg}".format(t="Regularizer", reg=self.regularizer.__name__))
            print(json.dumps(self.regularizer_params))
            self.regularizer(model, **self.regularizer_params)

            print("{t:<20}: {opt}".format(t="Optimizer", opt=self.optimizer.__name__))
            print(json.dumps(self.optimizer_params))
            optimizer = self.optimizer(**self.optimizer_params)

            print("{t:<20}: {loss}".format(t="Loss Function", loss=self.loss.__name__))

            # if self.multi_gpu > 1:
            #     print("Using %d GPUs for training" % self.multi_gpu)
            #     model = tf.keras.utils.multi_gpu_model(model, self.multi_gpu, cpu_merge=False)
            model.compile(optimizer, loss=self.loss, metrics=self.eval_metrics)

            print("{t:<20}: {filename}".format(t="Model checkpoint", filename=self.model_ckpt))

        return model

    def _cv(self):
        """ Cross validation """
        y_true = []
        y_pred = []

        tf.reset_default_graph()
        eval_model = self.model_generating_fn(training=False)

        eval_model.load_weights(os.path.join(self._get_save_dir(), "final.h5"), by_name=True)

        for i in range(len(self.val_generator)):
            image, label = self.val_generator[i]
            pred = eval_model.predict(image)
            pred = self.postprocessor.get_predition(pred)
            y_true.append(label)
            y_pred.append(pred)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        score = self.cv_fn(y_true, y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

        result = {
            self.cv_fn.__name__: score,
            "confusion_matrix": confusion_matrix.tolist()
        }
        print("{metric_name}: {score:.4f}".format(metric_name=self.cv_fn.__name__, score=score))
        return result

    def _quadratic_weighted_kappa(self):
        """ compute quadratic weighted kappa on validation set """
        y_true = []
        y_pred = []

        for i in range(len(self.val_generator)):
            image, label = self.val_generator[i]
            pred = self.model.predict(image)
            pred = self.postprocessor.get_predition(pred)
            y_true.append(label)
            y_pred.append(pred)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        score = sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
        print("Quadratic weighted kappa: %.4f" % score)

        return score

    def train(self):
        self._read_input()
        self.train_generator, self.val_generator = self._get_input_generator()
        self.model = self._build_model()

        self.model.fit_generator(self.train_generator, steps_per_epoch=len(self.train_generator),
                                 validation_data=self.val_generator, validation_steps=len(self.val_generator),
                                 epochs=self.num_epochs, callbacks=self._get_callback())

        if self.fine_tuning_layers:
            for layer in self.model.layers[:-self.fine_tuning_layers]:
                layer.trainable = True

        self.model.save_weights(os.path.join(self._get_save_dir(), "final.h5"))

        cv_result = self._cv()
        self.eval = cv_result

        self.write_config()

    def save(self):
        tf.contrib.saved_model.save_keras_model(self.model, self._get_save_dir())

