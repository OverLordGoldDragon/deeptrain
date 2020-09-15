# -*- coding: utf-8 -*-
import os
import sys
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(__file__)
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

import pytest

from time import time
from copy import deepcopy

from backend import AE_CONFIGS, BASEDIR, tempdir, notify, make_autoencoder
from backend import _init_session, _do_test_load, _get_test_names


#### CONFIGURE TESTING #######################################################
batch_size = 128
width, height = 28, 28
channels = 1
batch_shape = (batch_size, width, height, channels)
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image_lz4f')

DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    data_loader='numpy-lz4f',
    data_dtype='float64',
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    data_batch_shape=batch_shape,
    shuffle=True,
)
VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    data_loader='numpy-lz4f',
    data_dtype='float64',
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    data_batch_shape=batch_shape,
    shuffle=False,
)

tests_done = {}
CONFIGS = deepcopy(AE_CONFIGS)
CONFIGS['datagen'] = DATAGEN_CFG
CONFIGS['val_datagen'] = VAL_DATAGEN_CFG
CONFIGS['traingen']['epochs'] = 2

autoencoder = make_autoencoder(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_autoencoder)
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['datagen']['labels_path'] = None
        C['val_datagen']['labels_path'] = None
        tg = init_session(C, model=autoencoder)
        tg.train()
        _test_load(tg, C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


@notify(tests_done)
def test_predict():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen']['eval_fn'] = 'predict'
        tg = init_session(C, model=autoencoder)
        tg.train()
        _test_load(tg, C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
