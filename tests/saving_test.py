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
from pathlib import Path

from backend import CL_CONFIGS, AE_CONFIGS, tempdir, notify
from backend import make_classifier, make_autoencoder
from backend import K, load_model, _init_session, _get_test_names

from deeptrain.callbacks import VizAE2D

#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(CL_CONFIGS)
AE_CONFIGS = deepcopy(AE_CONFIGS)

classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None,
                 model_fn=make_classifier):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=model_fn)
###############################################################################

@notify(tests_done)
def test_model_save():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=classifier)

        for name in ('model:weights', 'optimizer_state'):
            if name in tg.saveskip_list:
                tg.saveskip_list.pop(tg.saveskip_list.index(name))
        if 'model' not in tg.saveskip_list:
            tg.saveskip_list.append('model')

        tg.train()
        _validate_save_load(tg, C)


@notify(tests_done)
def test_model_save_weights():
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=classifier)

        if 'model' in tg.saveskip_list:
            tg.saveskip_list.pop(tg.saveskip_list.index('model'))
        for name in ('model:weights', 'optimizer_state'):
            if name not in tg.saveskip_list:
                tg.saveskip_list.append(name)

        tg.train()
        _validate_save_load(tg, C)


def _validate_save_load(tg, C):
    def _get_load_path(tg, logdir):
        for postfix in ('weights', 'model', 'model_noopt'):
            postfix += '.h5'
            path = [str(p) for p in Path(logdir).iterdir()
                    if p.name.endswith(postfix)]
            if path:
                return path[0]
        raise Exception(f"no model save file found in {logdir}")

    # get behavior before saving, to ensure no changes presave-to-postload
    data = np.random.randn(*tg.model.input_shape)

    Wm_save = tg.model.get_weights()
    Wo_save = K.batch_get_value(tg.model.optimizer.weights)
    preds_save = tg.model.predict(data, batch_size=len(data))

    tg.checkpoint()
    logdir = tg.logdir
    tg.destroy(confirm=True)

    C['traingen']['logdir'] = logdir
    path = _get_load_path(tg, logdir)
    if path.endswith('weights.h5'):
        model = make_classifier(**C['model'])
        model.load_weights(path)
    else:
        model = load_model(path)
    tg = init_session(C, model=model)
    tg.load()

    Wm_load = tg.model.get_weights()
    Wo_load = K.batch_get_value(tg.model.optimizer.weights)
    preds_load = tg.model.predict(data, batch_size=len(data))

    for s, l in zip(Wm_save, Wm_load):
        assert np.allclose(s, l), "max absdiff: %s" % np.max(np.abs(s - l))
    for s, l in zip(Wo_save, Wo_load):
        assert np.allclose(s, l), "max absdiff: %s" % np.max(np.abs(s - l))
    assert np.allclose(preds_save, preds_load), (
          "max absdiff: %s" % np.max(np.abs(preds_save - preds_load)))


@notify(tests_done)
def test_warnings_and_exceptions():
    def _test_get_optimizer_state():
        C = deepcopy(CONFIGS)
        with tempdir(C['traingen']['logs_dir']), \
            tempdir(C['traingen']['best_models_dir']):
            tg = init_session(C, model=classifier)

            tg.optimizer_save_configs = {'exclude': ['updates']}
            tg._get_optimizer_state()

            tg.optimizer_save_configs = {'include': ['updates']}
            tg._get_optimizer_state()

    def _test_load_optimizer_state():
        C = deepcopy(CONFIGS)
        with tempdir(C['traingen']['logs_dir']), \
            tempdir(C['traingen']['best_models_dir']):
            tg = init_session(C, model=classifier)

            tg.optimizer_state = {}
            tg.optimizer_load_configs = {'exclude': ['updates']}
            tg._load_optimizer_state()

            tg.optimizer_state = {}
            tg.optimizer_load_configs = {'include': ['updates']}
            tg._load_optimizer_state()

    _test_get_optimizer_state()
    _test_load_optimizer_state()


@notify(tests_done)
def test_resumer_session_restart():
    """Ensures TrainGenerator can work with `model` recompiled on-the-fly,
    and load correctly from __init__.
    """
    C = deepcopy(AE_CONFIGS)
    C['traingen'].update(dict(
        epochs=4,
        key_metric='mae',
        eval_fn='predict',
        val_freq={'epoch': 2},
        plot_history_freq={'epoch': 2},
        unique_checkpoint_freq={'epoch': 2},
        model_save_kw=dict(include_optimizer=False, save_format='h5'),
        callbacks=[VizAE2D(n_images=8, save_images=True)],
    ))

    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model_fn=make_autoencoder)
        # do save optimizer weights & attrs to load later
        tg.saveskip_list.pop(tg.saveskip_list.index('optimizer_state'))

        tg.train()

        #### Phase 2 ##########
        tg.model.compile('nadam', 'mae')
        tg.epochs = 6
        tg.train()

        miscdir = os.path.join(tg.logdir, 'misc')
        assert [f"epoch{e}.png" in os.listdir(miscdir) for e in (2, 4, 6)]

        #### New session w/ changed model hyperparams ########################
        # get best save's model weights & TrainGenerator state
        best_weights = [str(p) for p in Path(tg.best_models_dir).iterdir()
                        if p.name.endswith('__weights.h5')]
        best_state   = [str(p) for p in Path(tg.best_models_dir).iterdir()
                        if p.name.endswith('__state.h5')]
        latest_best_weights = sorted(best_weights, key=os.path.getmtime)[-1]
        latest_best_state   = sorted(best_state,   key=os.path.getmtime)[-1]

        pre_load_epoch = tg.epoch
        pre_load_model_num = tg.model_num

        tg.destroy(confirm=True)
        del tg

        # change hyperparam
        C['model']['preout_dropout'] = .7
        # change loss
        C['model']['loss'] = 'mae'
        C['traingen']['epochs'] = 8
        C['traingen']['new_model_num'] = False
        # must re-instantiate callbacks object to hold new TrainGenerator
        C['traingen']['callbacks'] = [VizAE2D(n_images=8, save_images=True)]

        tg = init_session(C, loadpath=latest_best_state,
                          model_fn=make_autoencoder)
        tg.model.load_weights(latest_best_weights)

        assert tg.epoch == pre_load_epoch
        assert tg.model_num == pre_load_model_num + 1
        assert Path(tg.logdir).name.split('__')[0][1:] == str(tg.model_num)
        tg.train()
        assert tg.epoch > pre_load_epoch

        miscdir = os.path.join(tg.logdir, 'misc')
        assert 'epoch8.png' in os.listdir(miscdir)
        assert 'epoch6.png' not in os.listdir(miscdir)


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
