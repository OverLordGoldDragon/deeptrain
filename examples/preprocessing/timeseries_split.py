# -*- coding: utf-8 -*-
import os
import numpy as np
from pathlib import Path
from deeptrain.preprocessing import data_to_hdf5

batch_shape = (32, 20, 6)
n_batches = 8
overwrite = False

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "timeseries_split")

def make_dir_if_absent(_dir):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
###############################################################

if __name__ == '__main__':
    make_dir_if_absent(basedir)
    make_dir_if_absent(join(basedir, "train"))
    make_dir_if_absent(join(basedir, "val"))

    def make_data(batch_shape, n_batches):
        X = np.random.randn(n_batches, *batch_shape)
        Y = np.random.randint(0, 2, (n_batches, batch_shape[0], 1))
        return X, Y

    x_train, y_train = make_data(batch_shape, n_batches)
    x_test,  y_test  = make_data(batch_shape, n_batches // 2)

    kw = dict(batch_size=batch_shape[0], overwrite=overwrite)
    name = f"batch{batch_shape[0]}_"
    make = data_to_hdf5

    for i in range(n_batches):
        make(join(basedir, "train", f"{name}{i}.h5"), data=x_train[i:i+1], **kw)
        if i < len(x_test):
            make(join(basedir, "val", f"{name}{i}.h5"), data=x_test[i:i+1], **kw)

    data_to_hdf5(join(basedir, "train", "labels.h5"), data=y_train, **kw)
    data_to_hdf5(join(basedir, "val",   "labels.h5"), data=y_test,  **kw)
