    # -*- coding: utf-8 -*-
import os
import numpy as np
from pathlib import Path
from deeptrain.preprocessing import data_to_hdf5, numpy_data_to_numpy_sets

batch_size = 128
overwrite = True
labels_format = 'csv'  # 'h5' or 'csv'

###############################################################
join = lambda *args: str(Path(*args))
basedir = join(Path(__file__).parent, "data", "ptbdb")

def make_dir_if_absent(_dir):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)

def train_test_split(normal, abnormal, test_frac=.1):
    Nn = len(normal)
    Na = len(abnormal)
    n_train = normal[   int(Nn * test_frac):]
    a_train = abnormal[ int(Na * test_frac):]
    n_test  = normal[  :int(Nn * test_frac)]
    a_test  = abnormal[:int(Na * test_frac)]

    x_train = np.vstack([n_train, a_train])
    x_test  = np.vstack([n_test,  a_test])
    y_train = np.vstack([np.zeros((len(n_train), 1)),
                         np.ones( (len(a_train), 1))])
    y_test  = np.vstack([np.zeros((len(n_test), 1)),
                         np.ones( (len(a_test), 1))])
    assert len(x_train) == len(y_train)
    assert len(x_test)  == len(y_test)
    return (x_train, x_test), (y_train, y_test)

###############################################################

if __name__ == '__main__':
    # make dirs if absent
    make_dir_if_absent(basedir)
    make_dir_if_absent(join(basedir, "train"))
    make_dir_if_absent(join(basedir, "val"))

    # load data
    normal   = np.load(join(basedir, "normal.npy"))
    abnormal = np.load(join(basedir, "abnormal.npy"))

    # preprocess & split into sets
    normal   = np.expand_dims(normal, -1)
    abnormal = np.expand_dims(abnormal, -1)
    (x_train, x_test), (y_train, y_test) = train_test_split(normal, abnormal)
    x_train, y_train = numpy_data_to_numpy_sets(x_train, y_train,
                                                batch_size=batch_size,
                                                shuffle=True)
    x_test,  y_test  = numpy_data_to_numpy_sets(x_test,  y_test,
                                                batch_size=batch_size,
                                                shuffle=True)

    kw = dict(batch_size=batch_size, overwrite=overwrite, batches_dim0=True)
    data_to_hdf5(join(basedir, "train", "data.h5"), data=x_train, **kw)
    data_to_hdf5(join(basedir, "val",   "data.h5"), data=x_test,  **kw)

    data_to_hdf5(join(basedir, "train", "labels.h5"), data=y_train, **kw)
    data_to_hdf5(join(basedir, "val",   "labels.h5"), data=y_test,  **kw)
