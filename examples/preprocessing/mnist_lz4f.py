# -*- coding: utf-8 -*-
import os
import numpy as np
import keras.utils
from pathlib import Path
from tensorflow.keras.datasets import mnist
from deeptrain.preprocessing import numpy_data_to_numpy_sets, numpy_to_lz4f

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "mnist_lz4f")

def make_dir_if_absent(_dir):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
###############################################################

if __name__ == '__main__':
    make_dir_if_absent(basedir)
    make_dir_if_absent(join(basedir, "train"))
    make_dir_if_absent(join(basedir, "val"))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    y_train = keras.utils.to_categorical(y_train)
    x_test  = np.expand_dims(x_test,  -1)
    y_test  = keras.utils.to_categorical(y_test)

    axes = tuple(range(1, x_train.ndim))
    x_train = x_train / x_train.max(axis=axes, keepdims=True)
    x_test  = x_test  / x_test.max(axis=axes, keepdims=True)

    kw = dict(batch_size=128, data_basename='128batch', overwrite=True)
    numpy_data_to_numpy_sets(x_train, y_train, join(basedir, "train"), **kw)
    numpy_data_to_numpy_sets(x_test,  y_test,  join(basedir, "val"),   **kw)

    ###############################################################
    for dirname in ("train", "val"):
        for path in Path(join(basedir, dirname)).iterdir():
            if path.suffix == '.npy':
                numpy_to_lz4f(np.load(path), savepath=path, overwrite=True)
