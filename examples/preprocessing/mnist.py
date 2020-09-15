# -*- coding: utf-8 -*-
"""
Data is fed to TrainGenerator via DataGenerator. To work, data:
    - must be in one directory (or one file with all data)
    - file extensions must be same (.npy, .h5, etc)
    - file names must be enumerated with a common name (data1.npy, data2.npy, ...)
    - file batch size (# of samples, or dim 0 slices) should be same, but
      can also be in integer or fractal multiples of (x2, x3, x1/2, x1/3, ...)
    - labels must be in one file - unless feeding input as labels (e.g.
      autoencoder), which doesn't require labels files; just pass
      `TrainGenerator(input_as_labels=True)`.

This example:
    - Downloads MNIST & loads it with Kears' train-test split function
    - Creates directories to save processed data to
    - Splits up data into batch_size=128 arrays, and saves one array per file
      in format expected by DataGenerator
"""
import os
import numpy as np
import keras.utils
from pathlib import Path
from tensorflow.keras.datasets import mnist
from deeptrain.preprocessing import numpy_data_to_numpy_sets

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "mnist")

def make_dir_if_absent(_dir):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
###############################################################

if __name__ == '__main__':
    # will make folders only if they don't already exist
    make_dir_if_absent(basedir)
    make_dir_if_absent(join(basedir, "train"))
    make_dir_if_absent(join(basedir, "val"))

    # download & format MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train)
    x_test  = np.expand_dims(x_test,  -1)
    y_test  = keras.utils.to_categorical(y_test)

    # scale to [0, 1]
    axes = tuple(range(1, x_train.ndim))
    x_train = x_train / x_train.max(axis=axes, keepdims=True)
    x_test  = x_test  / x_test.max(axis=axes, keepdims=True)

    # make batch_size=128 arrays, one array per file
    # 60000 train samples make 468 batches; remainder is oversampled to make 469
    # 10000 test  samples, 78 batches -> 79 with oversampling
    # data saved as .npy files, labels as single .h5 file
    kw = dict(batch_size=128, data_basename='128batch')
    numpy_data_to_numpy_sets(join(basedir, "train"), x_train, y_train, **kw)
    numpy_data_to_numpy_sets(join(basedir, "val"),   x_test,  y_test,  **kw)
