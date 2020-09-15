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

from backend import BASEDIR, tempdir, notify, make_timeseries_classifier
from backend import _init_session, _do_test_load, _get_test_names
from deeptrain.callbacks import binary_preds_per_iteration_cb
from deeptrain.callbacks import binary_preds_distribution_cb
from deeptrain.callbacks import infer_train_hist_cb


#### CONFIGURE TESTING #######################################################
datadir = os.path.join(BASEDIR, 'tests', 'data', 'timeseries')
batch_size = 32

MODEL_CFG = dict(
    batch_shape=(batch_size, 4, 6),
    units=6,
    optimizer='adam',
    loss='binary_crossentropy'
)
DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.csv'),
    batch_size=batch_size,
    shuffle=True,
    preprocessor='timeseries',
    preprocessor_configs=dict(window_size=4),
)
VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    superbatch_path=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.csv'),
    batch_size=batch_size,
    shuffle=False,
    preprocessor='timeseries',
    preprocessor_configs=dict(window_size=4),
)
TRAINGEN_CFG = dict(
    epochs=2,
    reset_statefuls=True,
    max_is_best=False,
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
    best_subset_size=3,
    model_configs=MODEL_CFG,
)

CONFIGS = {'model': MODEL_CFG, 'datagen': DATAGEN_CFG,
           'val_datagen': VAL_DATAGEN_CFG, 'traingen': TRAINGEN_CFG}
tests_done = {}
model = make_timeseries_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_timeseries_classifier)
###############################################################################

@notify(tests_done)
def test_main():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        tg = init_session(C, model=model)
        tg.train()
        _test_load(tg, C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_weighted_slices():
    t0 = time()
    C = deepcopy(CONFIGS)
    C['traingen'].update(dict(eval_fn='predict',
                              loss_weighted_slices_range=(.5, 1.5),
                              pred_weighted_slices_range=(.5, 1.5)))
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=model)
        tg.train()
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_predict():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        C['traingen'].update(dict(
            eval_fn='predict',
            key_metric='f1_score',
            val_metrics=('loss', 'tnr', 'tpr'),
            plot_first_pane_max_vals=1,
            metric_printskip_configs={'val': 'f1_score'},
            dynamic_predict_threshold_min_max=(.35, .95),
            class_weights={0: 1, 1: 5},
            iter_verbosity=2,
            callbacks={'val_end': [infer_train_hist_cb,
                                   binary_preds_per_iteration_cb,
                                   binary_preds_distribution_cb]},
        ))
        tg = init_session(C, model=model)
        tg.train()
        _test_load(tg, C)
    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_start_increments():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), tempdir(
            C['traingen']['best_models_dir']):
        C['datagen']['preprocessor_configs'] = dict(window_size=4,
                                                    start_increments=[0, 2])
        C['traingen']['epochs'] = 2
        tg = init_session(C, model=model)
        tg.train()
        _test_load(tg, C)


@notify(tests_done)
def _test_load(tg, C):
    _do_test_load(tg, C, init_session)


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
