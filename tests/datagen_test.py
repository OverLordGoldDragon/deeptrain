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
import matplotlib.pyplot as plt
import contextlib, io

from copy import deepcopy

from backend import BASEDIR, tempdir, notify, _get_test_names
from deeptrain.util.misc import pass_on_error, argspec
from deeptrain.util.algorithms import ordered_shuffle
from deeptrain.util import TimeseriesPreprocessor
from deeptrain.util.data_loaders import DataLoader
from deeptrain import DataGenerator


datadir = os.path.join(BASEDIR, 'tests', 'data')

DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'image', 'train'),
    labels_path=os.path.join(datadir, 'image', 'train', 'labels.h5'),
    batch_size=128,
    shuffle=True,
)

tests_done = {}


@notify(tests_done)
def test_advance_batch():
    C = deepcopy(DATAGEN_CFG)
    C['superbatch_path'] = os.path.join(datadir, 'image', 'train')
    dg = DataGenerator(**C)
    dg.advance_batch()

    C['batch_size'] = 31
    dg = DataGenerator(**C)
    pass_on_error(dg.advance_batch)

    C['batch_size'] = 256
    dg = DataGenerator(**C)
    dg.set_nums_to_process = []
    pass_on_error(dg.advance_batch)

    C['data_loader'] = 'pigeon'
    pass_on_error(DataGenerator, **C)


@notify(tests_done)
def test_shuffle():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['superbatch_path'] = os.path.join(datadir, 'image', 'train')
    C['batch_size'] = 64
    dg = DataGenerator(**C)
    dg.preload_superbatch()
    dg.advance_batch()


@notify(tests_done)
def test_kwargs():
    C = deepcopy(DATAGEN_CFG)
    C['shuffle_group_batches'] = True
    C['shuffle_group_samples'] = True
    DataGenerator(**C)


@notify(tests_done)
def test_data_loader():
    def _test_auto_hdf5(C):
        dg = DataGenerator(**C)
        dg.advance_batch()

    def _test_hdf5(C):
        C['data_loader'] = 'hdf5'
        dg = DataGenerator(**C)
        dg.advance_batch()

    def _test_exceptions(C):
        C['data_loader'] = 'invalid_loader'
        pass_on_error(DataGenerator, **C)

    def _test_lz4f_dataset(C):
        del C['labels_path']
        C['data_path'] = os.path.join(datadir, 'image_lz4f', 'train',
                                      '128batch__1.npy')
        pass_on_error(DataGenerator, **C)

        C['data_loader'] = 'numpy-lz4f'
        pass_on_error(DataGenerator, **C)

        C['data_batch_shape'] = (128, 28, 28, 1)
        DataGenerator(**C)

    def _test_unknown(C):
        C['data_loader'] = lambda x: x
        C['data_path'] = os.path.join(datadir, 'image_lz4f', 'train',
                                      '128batch__1.npy')
        pass_on_error(DataGenerator, **C)

    def _test_validate_args(C):
        pass_on_error(DataLoader, 1, 1)

        kw = dict(path=C['data_path'], loader=1, filepaths=None)
        pass_on_error(DataLoader, **kw)

        kw['filepaths'] = ['x']
        pass_on_error(DataLoader, **kw)

    _C = deepcopy(DATAGEN_CFG)
    _C['data_path'] = os.path.join(datadir, 'timeseries_split', 'train')
    _C['labels_path'] = os.path.join(datadir, 'timeseries_split', 'train',
                                    'labels.h5')
    _C['batch_size'] = 128

    names, fns = zip(*locals().items())
    for name, fn in zip(names, fns):
        if hasattr(fn, '__code__') and argspec(fn)[0] == 'C':
            C = deepcopy(_C)
            fn(C)
            print("Passed", fn.__name__)


@notify(tests_done)
def test_labels_loaders():
    def _test_no_loader():
        C = deepcopy(DATAGEN_CFG)
        C['labels_loader'] = None
        C['labels_path'] = None
        DataGenerator(**C)

    _test_no_loader()


@notify(tests_done)
def test_preprocessors():
    def _test_uninstantiated(C):
        C['preprocessor'] = TimeseriesPreprocessor
        C['preprocessor_configs'] = dict(window_size=5)
        DataGenerator(**C)

    def _test_instantiated(C):
        TimeseriesPreprocessor(window_size=5)

    def _test_start_increment(C):
        pp = TimeseriesPreprocessor(window_size=25, start_increments=None)
        try:
            pp.start_increment = 5
            # shouldn't be able to set with start_increments = None
            assert False, ("shouldn't be able to set `start_increment`"
                            "with `start_increments == None`")
        except ValueError:
            pass

        pp = TimeseriesPreprocessor(window_size=25, start_increments=[0, 5])
        pp.start_increment = 5  # should throw a warning
        try:
            pp.start_increment = 5.0
            assert False, "shouldn't be able to set `start_increment` to a float"
        except ValueError:
            pass

    def _test_start_increment_warning(C):
        pp = TimeseriesPreprocessor(window_size=25, start_increments=[0, 5])

        str_io = io.StringIO()
        with contextlib.redirect_stdout(str_io):
            pp.start_increment = 4
        output = str_io.getvalue()
        assert "WARNING:" in output, "print(%s)" % output

    names, fns = zip(*locals().items())
    for name, fn in zip(names, fns):
        if name.startswith('_test_') or name.startswith('test_'):
            C = deepcopy(DATAGEN_CFG)
            fn(C)


@notify(tests_done)
def test_shuffle_group_batches():
    """Ensure reshape doesn't mix batch and spatial dimensions"""
    group_batch = np.random.randn(128, 28, 28, 1)
    labels = np.random.randint(0, 2, (128, 10))
    gb, lb = group_batch, labels

    batch_size = 64
    x0, x1 = gb[:64], gb[64:]
    y0, y1 = lb[:64], lb[64:]

    gb_shape, lb_shape = gb.shape, lb.shape
    gb = gb.reshape(-1, batch_size, *gb_shape[1:])
    lb = lb.reshape(-1, batch_size, *lb_shape[1:])
    x0adiff = np.sum(np.abs(gb[0] - x0))
    x1adiff = np.sum(np.abs(gb[1] - x1))
    y0adiff = np.sum(np.abs(lb[0] - y0))
    y1adiff = np.sum(np.abs(lb[1] - y1))
    assert x0adiff == 0, ("x0 absdiff: %s" % x0adiff)
    assert x1adiff == 0, ("x1 absdiff: %s" % x1adiff)
    assert y0adiff == 0, ("y0 absdiff: %s" % y0adiff)
    assert y1adiff == 0, ("y1 absdiff: %s" % y1adiff)

    gb, lb = ordered_shuffle(gb, lb)
    gb, lb = gb.reshape(*gb_shape), lb.reshape(*lb_shape)
    assert (gb.shape == gb_shape) and (lb.shape == lb_shape)


@notify(tests_done)
def test_infer_info():
    def _test_empty_data_path():
        C = deepcopy(DATAGEN_CFG)
        with tempdir() as dirpath:
            C['data_path'] = dirpath
            pass_on_error(DataGenerator, **C)

    def _test_no_supported_file_ext():
        C = deepcopy(DATAGEN_CFG)
        with tempdir() as dirpath:
            plt.plot([0, 1])
            plt.gcf().savefig(os.path.join(dirpath, "img.png"))
            C['data_path'] = dirpath
            pass_on_error(DataGenerator, **C)

    _test_empty_data_path()
    _test_no_supported_file_ext()


@notify(tests_done)
def test_warnings_and_exceptions():
    def _test_init():
        C = deepcopy(DATAGEN_CFG)
        C['superbatch_set_nums'] = 'all'
        C['superbatch_path'] = 'x'
        pass_on_error(DataGenerator, **C)

        C = deepcopy(DATAGEN_CFG)
        C['labels_path'] = 1
        pass_on_error(DataGenerator, **C)

        C['data_path'] = 1
        pass_on_error(DataGenerator, **C)

    def _test_misc():
        C = deepcopy(DATAGEN_CFG)
        dg = DataGenerator(**C)
        dg.superbatch = {'1': 1, '2': 2}
        dg.superbatch_set_nums = ['3']
        pass_on_error(dg._get_next_batch, set_num='3', warn=True)

        dg.all_labels = {}
        pass_on_error(dg._get_next_labels, set_num='3')

        pass_on_error(setattr, dg, 'load_data', 1)
        pass_on_error(setattr, dg, 'load_labels', 1)

        with tempdir() as dirpath:
            path = os.path.join(dirpath, "arr.npy")
            np.save(path, np.array([1]))
            C = deepcopy(DATAGEN_CFG)
            C['labels_path'] = None
            C['data_path'] = path
            pass_on_error(DataGenerator, **C)

    def _test_make_group_batch_and_labels():
        C = deepcopy(DATAGEN_CFG)
        dg = DataGenerator(**C)

        dg.batch = np.random.randn(128, 10)
        dg.labels = np.random.randn(129, 10)
        pass_on_error(dg._make_group_batch_and_labels, n_batches=2)

        dg.shuffle_group_samples = True
        dg.labels = dg.batch.copy()
        dg._make_group_batch_and_labels(n_batches=2)

        dg.labels_path = None
        dg._make_group_batch_and_labels(n_batches=2)

        dg.shuffle_group_batches = True
        dg.shuffle_group_samples = False
        dg._make_group_batch_and_labels(n_batches=2)

    def _test_infer_and_set_info():
        C = deepcopy(DATAGEN_CFG)
        with tempdir() as dirpath:
            path = os.path.join(dirpath, "arr.npy")
            np.save(path, np.array([1]))
            C['labels_path'] = None
            C['data_loader'] = DataLoader(path, loader='numpy')
            DataGenerator(**C)

            C['labels_loader'] = DataLoader(path, loader='numpy')
            DataGenerator(**C)

        C['data_loader'] = DataGenerator
        pass_on_error(DataGenerator, **C)

        C['labels_loader'] = None
        C['data_loader'] = DataLoader
        DataGenerator(**C)

        C['labels_loader'] = DataGenerator
        pass_on_error(DataGenerator, **C)

    _test_init()
    _test_misc()
    _test_make_group_batch_and_labels()
    _test_infer_and_set_info()


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
