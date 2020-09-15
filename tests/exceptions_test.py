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

from unittest import mock
from time import time
from copy import deepcopy

from backend import AE_CONFIGS, tempdir, notify
from backend import _init_session, _get_test_names
from backend import make_timeseries_classifier, make_autoencoder
from deeptrain import util
from deeptrain import metrics
from deeptrain import preprocessing
from deeptrain import introspection
from deeptrain import DataGenerator
from deeptrain.visuals import layer_hists
from deeptrain.util.misc import pass_on_error
from deeptrain.util import misc


#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(AE_CONFIGS)
batch_size, width, height, channels = CONFIGS['model']['batch_shape']

CL_CFG = dict(
    batch_shape=(batch_size, 25, 16),
    units=16,
    optimizer='adam',
    loss='binary_crossentropy'
)
AE_CFG = AE_CONFIGS['model']

classifier  = make_timeseries_classifier(**CL_CFG)
autoencoder = make_autoencoder(**AE_CFG)
init_session = _init_session

###############################################################################

@notify(tests_done)
def test_datagen():
    t0 = time()
    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=autoencoder)
        tg.train()

        dg = tg.datagen
        dg.advance_batch()
        dg.batch = dg.batch[:1]
        dg.batch_loaded = False
        pass_on_error(dg.advance_batch)
        dg.batch_loaded = True
        dg.advance_batch(forced=False)

        dg.shuffle = True
        dg.all_data_exhausted = True
        dg._validate_batch()

        dg.batch = []
        dg.batch_exhausted = True
        dg._validate_batch()

        dg.set_nums_to_process = dg.set_nums_original.copy()
        pass_on_error(dg._set_set_nums, ['99', '100'], ['100', '101'])
        pass_on_error(dg._set_set_nums, ['1', '2'], ['100', '101'])
        dg.superbatch_path = None
        pass_on_error(dg._set_set_nums, ['1', '2'], ['1', '2'])

        dg._set_preprocessor(None, {})
        pass_on_error(dg._set_preprocessor, "x", {})

        C['datagen']['invalid_kwarg'] = 5
        pass_on_error(DataGenerator, **C['datagen'])

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_visuals():
    def _layer_hists(model):
        pass_on_error(layer_hists, model, '*', mode='gradients')
        pass_on_error(layer_hists, model, '*', mode='outputs')
        pass_on_error(layer_hists, model, '*', mode='skeletons')

    C = deepcopy(CONFIGS)
    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=autoencoder)
        model = tg.model
        _layer_hists(model)


@notify(tests_done)
def test_util():
    t0 = time()

    def _util_make_autoencoder(C, new_model=False):
        C['model'] = AE_CFG
        C['traingen']['model_configs'] = AE_CFG
        C['traingen']['input_as_labels'] = True
        if new_model:
            return init_session(C, model_fn=make_autoencoder)
        else:
            autoencoder.loss = 'mse'  # reset changed configs
            return init_session(C, model=autoencoder)

    def _util_make_classifier(C, new_model=False):
        C['model'] = CL_CFG
        C['traingen']['model_configs'] = CL_CFG
        C['traingen']['input_as_labels'] = False
        if new_model:
            return init_session(C, model_fn=make_timeseries_classifier)
        else:
            classifier.loss = 'binary_crossentropy'  # reset changed configs
            return init_session(C, model=classifier)

    def _save_best_model(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.train()
        with mock.patch('os.remove') as mock_remove:
            mock_remove.side_effect = OSError('Permission Denied')
            tg.key_metric_history.append(-.5)  # ensure is new best
            tg._save_best_model(del_previous_best=True)
        with mock.patch('deeptrain.train_generator.TrainGenerator.save_report'
                   ) as mock_report:
            mock_report.side_effect = Exception()
            tg.key_metric_history.append(-1)  # ensure is new best
            tg._save_best_model()

    def checkpoint(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.train()
        tg.max_checkpoints = -1
        with mock.patch('os.remove') as mock_remove:
            mock_remove.side_effect = OSError('Permission Denied')
            tg.checkpoint(forced=True, overwrite=False)

        tg.logdir = None
        pass_on_error(tg.checkpoint)

    def save(C):  # [util.saving]
        tg = _util_make_autoencoder(C)
        tg.checkpoint()
        tg.model.loss = 'mean_squared_error'
        tg.train()
        tg.final_fig_dir = tg.logdir

        pass_on_error(tg.load)
        pass_on_error(tg.checkpoint, overwrite="underwrite")
        tg.datagen.set_nums_to_process = [9001]
        tg.save()
        tg._save_history_fig()
        tg._save_history_fig()
        tg.optimizer_load_configs = {'exclude': ['weights']}
        tg.loadskip_list = ['optimizer_load_configs']
        tg.datagen.loadskip_list = ['stuff']
        tg.load()

        tg._history_fig = 1
        tg._save_history_fig()

        tg.loadskip_list = 'auto'
        tg.load()
        tg.loadskip_list = 'none'
        tg.load()

        tg.optimizer_save_configs = {'include': []}
        tg.save()

        with mock.patch('backend.K.get_value') as mock_get_value:
            mock_get_value.side_effect = Exception()
            tg.save()

        tg.optimizer_save_configs = {'include': ['leaking_rate']}
        tg.datagen.group_batch = []
        with mock.patch('pickle.dump') as mock_dump:
            mock_dump.side_effect = Exception()
            tg.save()

        tg.logdir = 'abc'
        pass_on_error(tg.load)
        tg.logdir = None
        pass_on_error(tg.load)

    def get_sample_weight(C):  # [util.training]
        tg = _util_make_autoencoder(C)
        labels = np.random.randint(0, 2, (32, 3))
        tg.class_weights = {0: 1, 1: 2, 2: 3}
        tg.get_sample_weight(labels)

    def _get_api_metric_name(C):  # [util.training]
        util.training._get_api_metric_name(
            'accuracy', 'categorical_crossentropy')
        util.training._get_api_metric_name(
            'acc', 'sparse_categorical_crossentropy')
        util.training._get_api_metric_name('acc', 'binary_crossentropy')

    def _get_best_subset_val_history(C):  # [util.training]
        C['traingen']['best_subset_size'] = 2
        tg = _util_make_classifier(C)

        tg.val_datagen.slices_per_batch = 4
        tg._labels_cache = np.random.randint(0, 2, (3, 4, batch_size, 1))
        tg._preds_cache = np.random.uniform(0, 1, (3, 4, batch_size, 1))
        tg._sw_cache = np.random.randint(0, 2, (3, 4, batch_size, 1))
        tg._class_labels_cache = tg._labels_cache.copy()
        tg._val_set_name_cache = ['1', '2', '3']
        tg.key_metric = 'f1_score'
        tg.val_temp_history = {'f1_score': []}
        tg.key_metric_fn = metrics.f1_score
        tg._eval_fn_name = 'predict'
        tg.dynamic_predict_threshold_min_max = None

        tg._get_best_subset_val_history()

        tg._eval_fn_name = 'superfit'
        pass_on_error(tg._get_best_subset_val_history)

    def _update_temp_history(C):  # [util.training]
        tg = _util_make_classifier(C)

        tg.val_temp_history['loss'] = (1, 2, 3)
        tg._update_temp_history(metrics=(4,), val=True)
        tg.val_temp_history['loss'] = []
        tg._update_temp_history(metrics=(4,), val=True)

        tg.datagen.slice_idx = 1
        tg.datagen.slices_per_batch = 2
        tg.temp_history = {'binary_accuracy': []}
        tg.train_metrics = ['binary_accuracy']
        pass_on_error(tg._update_temp_history, metrics=[1], val=False)

        pass_on_error(tg._update_temp_history,
                      metrics=[dict(a=1, b=2)], val=False)

        # tg._update_temp_history([[1]], val=False)  # tests `_handle_non_scalar`

        tg.temp_history = {'f1_score': []}
        tg.train_metrics = ['f1_score']
        pass_on_error(tg._update_temp_history, metrics=[[1, 2]], val=False)

    def compute_gradient_norm(C):  # [introspection]
        pass_on_error(introspection.compute_gradient_norm, 0, 0, 0, mode="leftput")

    def _init_optimizer(C):  # [misc]
        tg = _util_make_classifier(C)
        tg.model.loss = 1
        pass_on_error(misc._init_optimizer, tg.model)

    def metrics_getattr(C):  # [TrainGenerator]
        def _raise(Except):
            raise Except()
        import_err = lambda: _raise(ImportError)

        with mock.patch('builtins.__import__', side_effect=import_err):
            pass_on_error(getattr, metrics, 'r2_score')

    def _validate_weighted_slices_range(C):  # [util.misc]
        C['traingen']['pred_weighted_slices_range'] = (.5, 1.5)
        C['traingen']['eval_fn'] = 'evaluate'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        tg = _util_make_autoencoder(C)
        tg.pred_weighted_slices_range = (.5, 1.5)
        tg._eval_fn_name = 'predict'
        tg.datagen.slices_per_batch = None
        tg.val_datagen.slices_per_batch = None
        pass_on_error(tg._validate_traingen_configs)

        C['traingen']['max_is_best'] = True
        C['traingen']['eval_fn'] = 'evaluate'
        C['traingen']['pred_weighted_slices_range'] = (.1, 1.1)
        pass_on_error(_util_make_classifier, C)

        C['traingen']['eval_fn'] = 'predict'
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['datagen'].pop('slices_per_batch', None)
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['traingen']['eval_fn'] = 'predict'
        tg = _util_make_classifier(C)
        tg.pred_weighted_slices_range = (.1, 1.1)
        tg.datagen.slices_per_batch = 1
        tg.val_datagen.slices_per_batch = 1
        tg._validate_traingen_configs()

    def _validate_metrics(C):  # [util.misc]
        C['traingen']['eval_fn'] = 'evaluate'
        C['traingen']['key_metric'] = 'hinge'
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['val_metrics'] = 'goblin'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        C['traingen']['key_metric'] = 'swordfish'
        C['traingen']['key_metric_fn'] = None
        C['traingen']['eval_fn'] = 'predict'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        C['traingen']['val_metrics'] = None
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['key_metric'] = 'loss'
        C['traingen']['max_is_best'] = True
        _util_make_autoencoder(C)

        C = deepcopy(CONFIGS)
        C['traingen']['eval_fn'] = 'predict'
        C['traingen']['val_metrics'] = 'cosine_similarity'
        pass_on_error(_util_make_autoencoder, C)

        C = deepcopy(CONFIGS)
        C['traingen']['eval_fn'] = 'predict'
        tg = _util_make_autoencoder(C)
        tg.model.loss = 'hl2'
        pass_on_error(tg._validate_traingen_configs)

        tg.train_metrics = ['tnr', 'tpr']
        tg.val_metrics = ['tnr', 'tpr']
        tg.key_metric = 'tnr'
        pass_on_error(tg._validate_traingen_configs)

    def _validate_directories(C):  # [util.misc]
        C['traingen']['best_models_dir'] = None
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['traingen']['logs_dir'] = None
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        C['traingen']['best_models_dir'] = None
        C['traingen']['logs_dir'] = None
        pass_on_error(_util_make_classifier, C)

    def _validate_optimizer_save_configs(C):  # [util.misc]
        C['traingen']['optimizer_save_configs'] = {
            'include': 'weights', 'exclude': 'updates'}
        pass_on_error(_util_make_classifier, C)

    def _validate_class_weights(C):  # [util.misc]
        C['traingen']['class_weights'] = {'0': 1, 1: 2}
        pass_on_error(_util_make_classifier, C)

        C['traingen']['class_weights'] = {0: 1}
        pass_on_error(_util_make_classifier, C)

        C = deepcopy(CONFIGS)
        tg = _util_make_classifier(C)
        tg.model.loss = 'categorical_crossentropy'
        tg.class_weights = {0: 1, 2: 5, 3: 6}
        tg._validate_traingen_configs()

    def _validate_best_subset_size(C):  # [util.misc]
        C['traingen']['best_subset_size'] = 5
        C['val_datagen']['shuffle_group_samples'] = True
        pass_on_error(_util_make_classifier, C)

    def _validate_metric_printskip_configs(C):  # [util.misc]
        C['traingen']['metric_printskip_configs'] = {'val': ('loss',)}
        _util_make_autoencoder(C)

    def _validate_savelist_and_metrics(C):  # [util.misc]
        C['traingen']['savelist'] = ['{labels}']
        C['traingen']['train_metrics'] = ('loss',)
        pass_on_error(_util_make_autoencoder, C)

    def _validate_loadskip_list(C):  # [util.misc]
        C['traingen']['loadskip_list'] = 'invalid'
        pass_on_error(_util_make_autoencoder, C)

    def _validate_callbacks(C):  # [util.misc]
        C['traingen']['callbacks'] = {'invalid_stage': 1}
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['callbacks'] = {'save': 1}
        pass_on_error(_util_make_autoencoder, C)

        C['traingen']['callbacks'] = 1
        pass_on_error(_util_make_autoencoder, C)

    def _validate_input_as_labels(C):  # [util.misc]
        C['traingen']['input_as_labels'] = False
        C['datagen']['labels_path'] = None
        C['val_datagen']['labels_path'] = None
        pass_on_error(_util_make_classifier, C)

    def _validate_model_save_kw(C):  # [util.misc]
        C['traingen']['model_save_kw'] = None
        C['traingen']['model_save_weights_kw'] = None
        _util_make_classifier(C)

    def _validate_freq_configs(C):  # [util.misc]
        C['traingen']['val_freq'] = 1
        pass_on_error(_util_make_classifier, C)

        C['traingen']['val_freq'] = {'epoch': 1, 'batch': 2}
        pass_on_error(_util_make_classifier, C)

    def _traingen_callbacks(C):  # [train_generator]
        tg = _util_make_autoencoder(C)
        tg.callbacks = [1]
        pass_on_error(tg._apply_callbacks)

        from deeptrain.callbacks import TraingenCallback
        tc = TraingenCallback()
        def raise_exception(self):
            raise NotImplementedError
        tc.init_with_traingen = raise_exception
        tg.callbacks = [tc]
        pass_on_error(tg._init_callbacks)

    def _on_val_end(C):  # [train_generator]
        tg = _util_make_autoencoder(C)
        tg.batch_size = 'x'
        pass_on_error(tg._on_val_end)

        tg.epochs += 1
        tg._train_loop_done = True

        def validate(self):
            self._train_loop_done = False
        tg.validate = validate.__get__(tg)
        pass_on_error(tg.train)

        tg._train_loop_done = False
        tg._fit_fn_name = 'x'
        tg.epochs += 1
        pass_on_error(tg.train)

        tg.batch_size = 'a'
        tg._eval_fn_name = 'predict'
        pass_on_error(tg._on_val_end, 0, 0, 0)

    def _train_postiter_processing(C):  # [train_generator]
        tg = _util_make_autoencoder(C)
        tg.datagen.batch_exhausted = True
        with mock.patch('deeptrain.train_generator.TrainGenerator.'
                        '_update_train_history') as mock_update:
            mock_update.side_effect = Exception
            tg._update_temp_history = lambda x: x
            pass_on_error(tg._train_postiter_processing, [])

    def _traingen_properties(C):  # [train_generator]
        tg = _util_make_autoencoder(C)
        pass_on_error(setattr ,tg, 'eval_fn', 1)

        pass_on_error(setattr, tg, 'eval_fn', tg.model.summary)

    def append_examples_dir_to_sys_path(C):  # [util.misc]
        util.misc.append_examples_dir_to_sys_path()

        with mock.patch('pathlib.Path.is_dir') as mock_dir:
            mock_dir.side_effect = lambda: False
            pass_on_error(util.misc.append_examples_dir_to_sys_path)

    names, fns = zip(*locals().items())
    for name, fn in zip(names, fns):
        if hasattr(fn, '__code__') and misc.argspec(fn)[0] == 'C':
            with tempdir(CONFIGS['traingen']['logs_dir']), \
                tempdir(CONFIGS['traingen']['best_models_dir']):
                C = deepcopy(CONFIGS)  # reset dict
                fn(C)
                print("Passed", fn.__name__)

    print("\nTime elapsed: {:.3f}".format(time() - t0))


@notify(tests_done)
def test_data_to_hdf5(monkeypatch):  # [deeptrain.preprocessing]
    """Dedicated test since it uses monkeypatch"""
    C = deepcopy(CONFIGS)
    # set preemptively in case data.h5 somehow found in dir
    monkeypatch.setattr('builtins.input', lambda x: 'y')

    with tempdir(C['traingen']['logs_dir']) as loaddir:
        with open(os.path.join(loaddir, "data.txt"), 'w') as txt:
            txt.write("etc")
        savepath = os.path.join(loaddir, "data.h5")
        pass_on_error(preprocessing.data_to_hdf5, savepath.replace('.h5', ''),
                      batch_size=32, loaddir=loaddir)

        data = np.random.randn(1, 32, 100)
        np.save(os.path.join(loaddir, "data.npy"), data)
        pass_on_error(preprocessing.data_to_hdf5, savepath=savepath,
                      batch_size=32, loaddir=loaddir)

        kw = dict(savepath=savepath, data=data, batch_size=32)
        pass_on_error(preprocessing.data_to_hdf5, **kw)

        os.remove(os.path.join(loaddir, "data.txt"))
        preprocessing.data_to_hdf5(**kw)

        monkeypatch.setattr('builtins.input', lambda x: 'y')
        preprocessing.data_to_hdf5(**kw)

        monkeypatch.setattr('builtins.input', lambda x: 'n')
        pass_on_error(preprocessing.data_to_hdf5, **kw)

        preprocessing.data_to_hdf5(overwrite=True, **kw)

        pass_on_error(preprocessing.data_to_hdf5, overwrite=False, **kw)

        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], loaddir=None, data=None)

        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], loaddir=loaddir, data=data)

        _data = [data[0], data[0, :31]]
        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], data=_data, overwrite=True)

        _data = [np.vstack([data[0], data[0]])]
        pass_on_error(preprocessing.data_to_hdf5, kw['savepath'],
                      kw['batch_size'], data=_data, overwrite=True)


@notify(tests_done)
def test_preprocessing():  # [deeptrain.preprocessing]
    data = np.random.randn(15, 2)
    pass_on_error(preprocessing.numpy2D_to_csv, data, batch_size=16)

    lz4f_cache = preprocessing.lz4f
    preprocessing.lz4f = None
    pass_on_error(preprocessing.numpy_to_lz4f, data)
    preprocessing.lz4f = lz4f_cache


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
