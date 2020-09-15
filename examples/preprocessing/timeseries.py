# -*- coding: utf-8 -*-
import os
import numpy as np
from pathlib import Path
from deeptrain.preprocessing import data_to_hdf5, numpy2D_to_csv


batch_shape = (16, 20, 6)
n_batches = 16
save_batch_size = 32
overwrite = False
labels_format = 'csv'  # 'h5' or 'csv'

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "timeseries")

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

    kw = dict(batch_size=save_batch_size, overwrite=overwrite)
    data_to_hdf5(join(basedir, "train", "data.h5"), data=x_train, **kw)
    data_to_hdf5(join(basedir, "val",   "data.h5"), data=x_test,  **kw)

    if labels_format == 'h5':
        data_to_hdf5(join(basedir, "train", "labels.h5"), data=y_train, **kw)
        data_to_hdf5(join(basedir, "val",   "labels.h5"), data=y_test,  **kw)
    elif labels_format == 'csv':
        kw.pop('overwrite')
        y_train, y_test = y_train.squeeze(axis=-1).T, y_test.squeeze(axis=-1).T
        numpy2D_to_csv(y_train, join(basedir, "train", "labels.csv"), **kw)
        numpy2D_to_csv(y_test,  join(basedir, "val",   "labels.csv"), **kw)
    else:
        raise ValueError("unsupported `labels_format`: " + labels_format + "; "
                         "supported are: h5, csv")
