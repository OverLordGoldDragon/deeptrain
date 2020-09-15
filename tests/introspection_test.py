# -*- coding: utf-8 -*-
import os
import sys
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(__file__)
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

import pytest
import numpy as np

from copy import deepcopy

from backend import Adam
from backend import AE_CONFIGS, notify, make_autoencoder
from backend import _init_session, _get_test_names
from deeptrain import introspection


#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(AE_CONFIGS)
del CONFIGS['traingen']['logs_dir']         # unused
del CONFIGS['traingen']['best_models_dir']  # unused

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                          model=model, model_fn=make_autoencoder)

_tg = init_session(CONFIGS)  # save time on redundant re-init's
_tg.train()
###############################################################################

@notify(tests_done)
def test_gather_over_dataset():
    _tg.gradient_norm_over_dataset(n_iters=None, prog_freq=3)
    _tg.gradient_norm_over_dataset(n_iters=None, prog_freq=3, norm_fn=np.abs)
    _tg.gradient_sum_over_dataset(n_iters=5, prog_freq=3)

    x, y, sw = _tg.get_data()
    _tg.compute_gradient_norm(x, y, sw)  # not gather, but test anyway


@notify(tests_done)
def test_print_dead_nan():
    def _test_print_nan_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e9)
        tg = init_session(C)
        tg.train()
        tg.check_health()

    def _test_print_dead_weights():
        C = deepcopy(CONFIGS)
        C['model']['optimizer'] = Adam(lr=1e-4)
        tg = init_session(C)
        tg.train()
        tg.check_health(dead_threshold=.1)
        tg.check_health(notify_detected_only=False)
        tg.check_health(notify_detected_only=False, dead_threshold=.5,
                        dead_notify_above_frac=2)

    _test_print_nan_weights()
    _test_print_dead_weights()


@notify(tests_done)
def test_compute_gradient_norm():
    dg = _tg.datagen
    _tg.compute_gradient_norm(dg.batch, dg.batch, scope='global', norm_fn=np.abs)


@notify(tests_done)
def test_grads_fn():
    dg = _tg.datagen
    grads_fn = introspection._make_gradients_fn(_tg.model, 0, mode="outputs")
    _ = grads_fn(dg.batch, dg.batch, sw=None)
    _ = grads_fn([dg.batch], [dg.batch], sw=None)


@notify(tests_done)
def test_info_and_interrupt_status():
    _tg.info()

    _tg._train_postiter_processed = False
    _tg._train_loop_done = True
    _tg.interrupt_status()

    _tg._train_loop_done = False
    _tg.interrupt_status()

    _tg._val_loop_done = True
    _tg._train_loop_done = False
    _tg.interrupt_status()

    _tg._train_loop_done = True
    _tg._val_postiter_processed = True
    _tg.interrupt_status()

    _tg._val_postiter_processed = False
    _tg.interrupt_status()

    _tg._val_loop_done = False
    _tg.interrupt_status()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
