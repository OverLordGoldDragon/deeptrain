# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
# ensure `tests` & package directory paths are on top of Python's module search
filedir = os.path.dirname(__file__)
sys.path.insert(0, str(Path(filedir).parent))
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

import pytest
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from types import LambdaType

from backend import BASEDIR, tempdir, notify, ModelDummy, TraingenDummy
from backend import _get_test_names
from backend import Adam
from deeptrain.util import searching
from deeptrain.util import misc
from deeptrain.util import logging
from deeptrain.util import training
from deeptrain.util import configs
from deeptrain.util import _default_configs
from deeptrain.util.misc import pass_on_error, _validate_traingen_configs
from deeptrain import metrics
from deeptrain import preprocessing
from tests import _methods_dummy  # <- need package directory on top for this


tests_done = {}


@notify(tests_done)
def test_searching():
    labels = np.random.randint(0, 2, (32,))
    preds = np.random.uniform(0, 1, (32,))
    metric_fn = metrics.f1_score
    searching.find_best_predict_threshold(labels, preds, metric_fn, verbosity=2)


@notify(tests_done)
def test_misc():
    def _test_init_optimizer():
        model = ModelDummy()
        model.loss = 'sparse_categorical_crossentropy'
        misc._init_optimizer(model)
        model.loss = 'hinge'
        misc._init_optimizer(model)
        model.loss = 'poisson'
        misc._init_optimizer(model)
        model.loss = 'categorical_crossentropy'
        misc._init_optimizer(model, class_weights={0: 1, 1: 5})
        misc._init_optimizer(model, class_weights=None)
        model.loss = 'goblin'
        pass_on_error(misc._init_optimizer, model)

    def _test_make_plot_configs_from_metrics():
        tg = TraingenDummy()
        tg.train_metrics = ['binary_crossentropy', 'hinge']
        tg.val_metrics = ['loss', 'f1_score', 'tnr', 'tpr', 'kld']
        tg.plot_first_pane_max_vals = 1
        tg.key_metric = 'loss'
        misc._make_plot_configs_from_metrics(tg)
        tg.key_metric = 'f1_score'
        misc._make_plot_configs_from_metrics(tg)

    def _test_get_module_methods():
        mm = misc.get_module_methods(_methods_dummy)
        assert len(mm) == 2 and 'fn1' in mm and 'not_lambda' in mm
        assert isinstance(mm['fn1'], LambdaType)
        assert isinstance(mm['not_lambda'], LambdaType)

    def _test_capture_args():
        class A():
            @misc.capture_args
            def __init__(self, a, b, c, *args, d=1, e=2, **kwargs):
                self._passed_args = kwargs.pop('_passed_args', None)

        a = A(1, 2, 3, 4, d=5, g=6)
        args = a._passed_args
        assert all(arg in args for arg in ('a', 'b', 'c', '*args', 'g'))
        assert args['a'] == 1
        assert args['b'] == 2
        assert args['c'] == 3
        assert args['*args'] == (4,)
        assert args['g'] == 6

    def _test_try_except():
        misc.try_except(lambda: 1 / 0, lambda: 1)

    # def _test_extract_pickleable():  # moved to experimental
    #     class Dummy():               # keep for future reference
    #         def __init__(self):
    #             self.a = 1
    #             self.arr = np.random.randn(2, 2)
    #     _ = misc.extract_pickleable(vars(Dummy()))

    for name, fn in locals().items():
        if name.startswith('_test_'):
            fn()


@notify(tests_done)
def test_logging():
    def _testget_unique_model_name():
        tg = TraingenDummy()
        tg.model_name_configs = {'datagen.shuffle': ''}
        os.mkdir(os.path.join(tg.logs_dir, 'M0'))
        logging.get_unique_model_name(tg)

    def _test_log_init_state():
        class SUV():
            pass

        tg = TraingenDummy()
        tg.SUV = SUV()
        logging._log_init_state(tg, source_lognames=['swordfish', 'SUV'])
        logging._log_init_state(tg, source_lognames='*', verbose=1)
        logging._log_init_state(tg, source_lognames=None)

    def _test_generate_report():
        tg = TraingenDummy()
        tg.report_configs = {'model': dict(stuff='staff'), 'saitama': None}
        pass_on_error(logging.generate_report, tg)

        tg.report_configs = {'model': {'genos': [1]}}
        pass_on_error(logging.generate_report, tg)

        tg.report_configs = {'model': {'include': [], 'exclude': []}}
        pass_on_error(logging.generate_report, tg)

        tg.report_configs = {'model': {'exclude_types': ['etc']}}
        tg.model_configs = {'a': 1}
        pass_on_error(logging.generate_report, tg)

    def _test_save_report():
        tg = TraingenDummy()
        tg.report_configs = {'model': {'include': []}}
        tg.logdir = ''
        pass_on_error(logging.save_report, tg, '')

    logs_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_logs')
    best_models_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_models')
    with tempdir(logs_dir), tempdir(best_models_dir):
        _testget_unique_model_name()

    _test_log_init_state()
    _test_generate_report()
    _test_save_report()


@notify(tests_done)
def test_training():
    def _test_weighted_normalize_preds():
        tg = TraingenDummy()
        tg.val_datagen.slices_per_batch = 1
        preds_all = np.random.uniform(0, 1, (3, 1, 16, 100))
        training._weighted_normalize_preds(tg, preds_all)

    def _test_validate_data_shapes():
        tg = TraingenDummy()
        tg.model.batch_size = None
        tg.batch_size = 16
        tg.model.output_shape = (None, 100)
        data = {'preds_all': np.random.uniform(0, 1, (3, 16, 100))}
        training._validate_data_shapes(tg, data)

        data = {'preds_all': np.random.uniform(0, 1, (3, 1, 16, 100))}
        pass_on_error(training._validate_data_shapes, tg, data)

    def _test_validate_class_data_shapes():
        tg = TraingenDummy()
        tg.model.batch_size = None
        tg.batch_size = 16
        tg.model.output_shape = (None, 100)
        data = {'class_labels_all': np.random.uniform(0, 1, (3, 16, 100))}
        training._validate_class_data_shapes(tg, data, validate_n_slices=True)

    _test_weighted_normalize_preds()
    _test_validate_data_shapes()
    _test_validate_class_data_shapes()


@notify(tests_done)
def test_configs():
    for name_fn in (configs.NAME_PROCESS_KEY_FN,
                    _default_configs._DEFAULT_NAME_PROCESS_KEY_FN):
        cfg = dict(init_lr=[2e-4, 2e-4, 2e-4, 1e-4],
                   eta_t=(.9, 1.1, 2),
                   timesteps=13500,
                   name='leaf',
                   optimizer=Adam,
                   opt=Adam(),  # instantiated may lack __name__, must still pass
                   a=.5,
                   best_key_metric=0.91,
                   )
        assert name_fn('a', 'a', cfg) == '-a.5'
        assert name_fn('init_lr',   'lr',   cfg) == '-lr2e-4x3_1e-4'
        assert name_fn('eta_t',     'et',   cfg) == '-et.9_1.1_2'
        assert name_fn('timesteps', '',     cfg) == '-13.5k'
        assert name_fn('name',      'name', cfg) == '-name'
        assert name_fn('optimizer', '',     cfg) == '-Adam'
        assert name_fn('opt',       '',     cfg) == '-Adam'
        assert name_fn('best_key_metric', '__max', cfg) == '__max.910'

    names = ['PLOT_CFG', 'MODEL_NAME_CFG', 'REPORT_CFG',
             'TRAINGEN_SAVESKIP_LIST', 'TRAINGEN_LOADSKIP_LIST',
             'DATAGEN_SAVESKIP_LIST', 'DATAGEN_LOADSKIP_LIST',
             'METRIC_PRINTSKIP_CFG', 'METRIC_TO_ALIAS', 'ALIAS_TO_METRIC',
             'TRAINGEN_CFG', 'DATAGEN_CFG']
    for config in (configs, _default_configs):
        for name in names:
            name = [x for x in vars(config) if name in x][0]
            _ = getattr(config, name)


@notify(tests_done)
def test_preprocessing(monkeypatch):
    def _test_numpy_data_to_numpy_sets(datadir):
        with tempdir(datadir):
            data = np.random.randn(161, 2)
            labels = np.random.randint(0, 2, (161,))
            preprocessing.numpy_data_to_numpy_sets(
                data, labels, datadir, batch_size=32, shuffle=True,
                data_basename='ex', oversample_remainder=True)

            paths = [str(x) for x in Path(datadir).iterdir() if
                     x.suffix == '.npy']
            assert (len(paths) == 6), ("%s paths" % len(paths))  # 160 / 32

        os.mkdir(datadir)
        data = np.random.randn(161, 2)
        labels = np.random.randint(0, 2, (161,))

        preprocessing.numpy_data_to_numpy_sets(
            data, labels, datadir, batch_size=32, shuffle=True,
            data_basename='ex', oversample_remainder=False)
        os.remove(os.path.join(datadir, "labels.h5"))

        paths = [str(x) for x in Path(datadir).iterdir() if
                 x.suffix == '.npy']
        assert (len(paths) == 5), ("%s paths" % len(paths))  # 160 / 32

        return paths

    def _test_numpy_to_lz4f(datadir, paths):
        data = np.load(paths[0])
        savepath = os.path.join(datadir, "data_lz4f")
        preprocessing.numpy_to_lz4f(data, savepath=savepath)
        os.remove(savepath + '.npy')  # will error in to_hdf5

    def _test_data_to_hdf5(datadir, paths):
        X = np.array([np.load(path) for path in paths])
        kw = dict(savepath=os.path.join(datadir, "data.h5"), batch_size=32,
                  shuffle=True, compression='lzf', overwrite=None)

        monkeypatch.setattr('builtins.input', lambda x: "y")
        preprocessing.data_to_hdf5(loaddir=datadir, **kw)
        preprocessing.data_to_hdf5(data=X, **kw)

        kw.update(**dict(overwrite=True, load_fn=lambda x: x))
        preprocessing.data_to_hdf5(data=X, **kw)

    def _test_numpy2D_to_csv(datadir):
        def _batches_from_df(savepath):
            df = pd.read_csv(savepath)
            assert df.shape == (32, 2)
            x0 = df['0'][:16].to_numpy()
            x1 = df['0'][16:].to_numpy()
            return x0, x1

        def _test_sample_dim_0(savepath):
            X = np.random.randint(0, 2, (16, 4))
            preprocessing.numpy2D_to_csv(data=X, savepath=savepath,
                                         batch_size=32, sample_dim=0)
            x0, x1 = _batches_from_df(savepath)
            assert np.sum(np.abs(x0 - X[:, 0])) == 0
            assert np.sum(np.abs(x1 - X[:, 1])) == 0

        def _test_sample_dim_1(savepath):
            X = np.random.randint(0, 2, (4, 16))
            preprocessing.numpy2D_to_csv(data=X, savepath=savepath,
                                         batch_size=32, sample_dim=1)
            x0, x1 = _batches_from_df(savepath)
            assert np.sum(np.abs(x0 - X[0])) == 0
            assert np.sum(np.abs(x1 - X[1])) == 0

        os.mkdir(datadir)
        savepath = os.path.join(datadir, "labels.csv")
        _test_sample_dim_0(savepath)
        _test_sample_dim_1(savepath)

    datadir = os.path.join(BASEDIR, "_data")

    paths = _test_numpy_data_to_numpy_sets(datadir)
    _test_numpy_to_lz4f(datadir, paths)
    _test_data_to_hdf5(datadir, paths)
    shutil.rmtree(datadir)

    _test_numpy2D_to_csv(datadir)
    shutil.rmtree(datadir)


@notify(tests_done)
def test_validate_traingen_configs():
    def _validate_key_metric_fn(tg):
        tg.custom_metrics = 1
        tg.key_metric_fn = None
        tg._eval_fn_name = 'predict'
        pass_on_error(_validate_traingen_configs, tg)

        tg.key_metric_fn = 1
        tg._eval_fn_name = 'evaluate'
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_metrics(tg):
        tg.val_metrics = ['acc', 'loss']
        pass_on_error(_validate_traingen_configs, tg)

        tg.key_metric_fn = None
        tg.custom_metrics = {}
        tg._eval_fn_name = 'predict'
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_best_subset_size(tg):
        tg.best_subset_size = 'a'
        pass_on_error(_validate_traingen_configs, tg)

        tg.best_subset_size = 0
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_dynamic_predict_threshold_min_max(tg):
        tg.val_metrics = ['loss']
        tg.key_metric = 'loss'
        tg.key_metric_fn = None
        tg.dynamic_predict_threshold_min_max = (.3, .8)
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_or_make_plot_configs(tg):
        tg.plot_configs = ['1']
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_model_save_kw(tg):
        tg.model_save_kw = None
        tg.model_save_weights_kw = None
        pass_on_error(_validate_traingen_configs, tg)

    def _validate_freq_configs(tg):
        tg.val_freq = 1
        pass_on_error(_validate_traingen_configs, tg)

        tg.val_freq = {'epoch': 1, 'batch': 2}
        pass_on_error(_validate_traingen_configs, tg)

    loc_names = list(locals())
    for name in loc_names:
        if name.startswith('_validate_'):
            tg = TraingenDummy()
            locals()[name](tg)


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
