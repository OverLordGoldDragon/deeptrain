# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
   - Setting custom loss and metrics
   - Setting custom data loader and preprocessor
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()
from utils import make_autoencoder, init_session
from utils import AE_CONFIGS as C

import numpy as np
from tensorflow.keras import backend as K
from deeptrain.metrics import _weighted_loss, _standardize
from deeptrain.util.preprocessors import Preprocessor

#%%# Make customs ###########################################################
def mean_L_error(y_true, y_pred, sample_weight=1):
    L = 1.5  # configurable
    y_true, y_pred, sample_weight = _standardize(y_true, y_pred, sample_weight)
    return _weighted_loss(np.mean(np.abs(y_true - y_pred) ** L, axis=-1),
                          sample_weight)

def mLe(y_true, y_pred):
    L = 1.5  # configurable
    return K.mean(K.pow(K.abs(y_true - y_pred), L), axis=-1)

def numpy_loader(self, set_num):
    # allow_pickle is irrelevant here, just for demo
    return np.load(self._path(set_num), allow_pickle=True)

class RandCropPreprocessor(Preprocessor):
    """2D random crop. MNIST is 28x28, we try 25x25 crops, e.g. batch[2:27, 3:28].
    """
    def __init__(self, size, crop_batch=True, crop_labels=False, crop_same=False):
        # length          -> (length, length)
        # (width, height) -> (width, height)
        assert isinstance(size, (tuple, int))
        self.size = size if isinstance(size, tuple) else (size, size)

        self.crop_batch = crop_batch
        self.crop_labels = crop_labels
        self.crop_same = crop_same

    def process(self, batch, labels):
        if self.crop_batch:
            (x_start, x_end), (y_start, y_end) = self._make_crop_mask(batch)
            batch = batch[:, x_start:x_end, y_start:y_end]
        if self.crop_labels:
            if not self.crop_same or not self.crop_batch:
                (x_start, x_end), (y_start, y_end) = self._make_crop_mask(labels)
            labels = labels[:, x_start:x_end, y_start:y_end]
        return batch, labels

    def _make_crop_mask(self, data):
        _, w, h, *_ = data.shape  # (samples, width, height, channels)
        x_offset = np.random.randint(0, w - self.size[0])
        y_offset = np.random.randint(0, h - self.size[1])
        x_start, x_end = x_offset, x_offset + self.size[0]
        y_start, y_end = y_offset, y_offset + self.size[1]
        return (x_start, x_end), (y_start, y_end)

#%%# Update configs #########################################################
C['model'].update({'loss': mLe,
                   'batch_shape': (128, 24, 24, 1)})
C['datagen'].update({'data_loader': numpy_loader,
                     'preprocessor': RandCropPreprocessor(size=24)})
C['val_datagen'].update({'data_loader': numpy_loader,
                         'preprocessor': RandCropPreprocessor(size=24)})
C['traingen']['custom_metrics'] = {'mLe': mean_L_error}
tg = init_session(C, make_autoencoder)

#%%# Train ##################################################################
tg.train()
