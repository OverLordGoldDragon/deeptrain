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
import sklearn.metrics

from backend import notify, TraingenDummy, _get_test_names
from deeptrain.util.training import _get_val_history, _weighted_normalize_preds
from deeptrain.metrics import (
    f1_score,
    f1_score_multi_th,
    tnr,
    tpr,
    tnr_tpr,
    binary_informedness,
    roc_auc_score
    )
from deeptrain import metrics as metric_fns

tests_done = {}


@notify(tests_done)
def test_f1_score():
    def _test_basic():
        y_true = [0,     0,   1,   0,   0, 0, 1, 1]
        y_pred = [.01, .93, .42, .61, .15, 0, 1, .5]
        assert abs(f1_score(y_true, y_pred) - 1 / 3) < 1e-15

    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .65, .7, .8]
        assert f1_score(y_true, y_pred) == 0.5

    def _test_no_positive_predictions():
        y_true = [0, 0, 1]
        y_pred = [0, 0, 0]
        assert f1_score(y_true, y_pred) == 0

    def _test_vs_sklearn():
        y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
        np.random.shuffle(y_true)
        y_pred = np.random.uniform(0, 1, 32)

        test_score = f1_score(y_true, y_pred, pred_threshold=.5)
        sklearn_score = sklearn.metrics.f1_score(y_true, y_pred > .5)
        adiff = abs(test_score - sklearn_score)
        assert (adiff < 1e-10), ("sklearn: {:.15f}\ntest:    {:.15f}"
                                "\nabsdiff: {}".format(
                                    sklearn_score, test_score, adiff))
    _test_basic()
    _test_no_positive_labels()
    _test_no_positive_predictions()
    _test_vs_sklearn()


@notify(tests_done)
def test_f1_score_multi_th():
    def _test_no_positive_labels():
        y_true = [0] * 6
        y_pred = [.1, .2, .3, .65, .7, .8]
        pred_thresholds = [.4, .6]
        assert np.all(f1_score_multi_th(y_true, y_pred, pred_thresholds)
                      == [.5, .5])

    def _test_nan_handling():
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0]
        pred_thresholds = [.4, .6]
        assert np.all(f1_score_multi_th(y_true, y_pred, pred_thresholds) == 0)

    def _compare_against_f1_score():
        y_true = np.random.randint(0, 2, (64,))
        y_pred = np.random.uniform(0, 1, (64,))
        pred_thresholds = [.01, .05, .1, .2, .4, .5, .6, .8, .95, .99]

        single_scores = [f1_score(y_true, y_pred, th) for th in pred_thresholds]
        parallel_scores = f1_score_multi_th(y_true, y_pred, pred_thresholds)
        assert np.allclose(single_scores, parallel_scores, atol=1e-15)

    _test_no_positive_labels()
    _test_nan_handling()
    _compare_against_f1_score()


@notify(tests_done)
def test_binaries():
    y_true = [0,  0,  0,  0,  1,  1,  1,  1]
    y_pred = [0, .6, .7, .9,  1,  0, .8, .6]

    assert tnr(y_true, y_pred) == .25
    assert tpr(y_true, y_pred) == .75
    assert tnr_tpr(y_true, y_pred) == [.25, .75]
    assert binary_informedness(y_true, y_pred) == 0.


@notify(tests_done)
def test_binaries_multi_th():
    def _compare_against_single_th(metric_name):
        y_true = np.random.randint(0, 2, (64,))
        y_pred = np.random.uniform(0, 1, (64,))
        pred_thresholds = [.01, .05, .1, .2, .4, .5, .6, .8, .95, .99]

        metric_fn = getattr(metric_fns, metric_name)
        metric_multi_th_fn = getattr(metric_fns, metric_name + '_multi_th')

        looped_scores = np.array([metric_fn(y_true, y_pred, th)
                                  for th in pred_thresholds])
        parallel_scores = metric_multi_th_fn(y_true, y_pred, pred_thresholds)
        if looped_scores.ndim > 1:  # tnr_tpr
            looped_scores = looped_scores.T

        assert np.allclose(looped_scores, parallel_scores), (
            f"'{metric_name}'\n"
            f"{np.vstack([looped_scores, parallel_scores]).T}")

    to_test = ['binary_accuracy', 'tnr', 'tpr', 'tnr_tpr', 'binary_informedness']
    for metric_name in to_test:
        _compare_against_single_th(metric_name)


@notify(tests_done)
def test_roc_auc_score():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    np.random.shuffle(y_true)
    y_pred = np.random.uniform(0, 1, 32)

    test_score = roc_auc_score(y_true, y_pred)
    sklearn_score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    adiff = abs(test_score - sklearn_score)
    assert (adiff < 1e-10), ("sklearn: {:.15f}\ntest:    {:.15f}"
                             "\nabsdiff: {}".format(
                                 sklearn_score, test_score, adiff))


@notify(tests_done)
def test_sample_unrolling():
    """Compare results from internal (TrainGenerator) data transforms vs. those
    of correct transforms done explicitly. Formats included:
        - (samples, outs[1:])
        - (slices, samples, outs[1:])
        - (batches, samples, outs[1:])
        - (batches, slices, samples, outs[1:])
        - Weighted & unweighted slices
        - Linear & nonlinear metrics

    outs == model.output_shape
    'Nonlinear' metrics == cannot be simple-averaged across samples, i.e.
    [fn(samples1) + fn(samples2)] / 2 != fn([samples1, samples2])
    """
    def _test_unrolling(fn):
        def unrolling_test(label_dim, tg, compare_fn):
            y_true, y_pred, tg, compare_fn = fn(label_dim, tg, compare_fn)
            m = _get_val_history(tg, for_current_iter=False)
            yt, yp = y_true.ravel(), y_pred.ravel()
            compare_fn(m, yt, yp, fn.__name__)
        return unrolling_test

    @_test_unrolling
    def _batches(label_dim, tg, compare_fn):
        y_true = np.random.randint(0, 2, (2, 8, label_dim))
        y_pred = np.random.uniform(0, 1, (2, 8, label_dim))

        tg.set_cache(y_true, y_pred)
        return y_true, y_pred, tg, compare_fn

    @_test_unrolling
    def _unweighted_slices(label_dim, tg, compare_fn):
        # unfixed labels across samples also tests fixed case
        y_true = np.random.randint(0, 2, (4, 8, label_dim))
        y_pred = np.random.uniform(0, 1, (4, 8, label_dim))

        tg.set_cache(y_true, y_pred)
        tg.val_datagen.slices_per_batch = y_true.shape[0]
        return y_true, y_pred, tg, compare_fn

    @_test_unrolling
    def _weighted_slices(label_dim, tg, compare_fn):
        # unfixed labels invalid for weighted slices since preds are normed
        y_true = np.random.randint(0, 2, (2, 8, label_dim))
        y_pred = np.random.uniform(0, 1, (4, 8, label_dim))
        y_true = np.vstack([y_true, y_true])  # labels fixed along slices

        tg.set_cache(y_true, y_pred)
        tg.val_datagen.slices_per_batch = y_true.shape[0]
        tg.loss_weighted_slices_range = (.2, 1.8)
        tg.pred_weighted_slices_range = (.2, 1.8)

        y_pred = np.expand_dims(y_pred, 0)
        y_pred = _weighted_normalize_preds(tg, y_pred)
        y_true = y_true[0]
        return y_true, y_pred, tg, compare_fn

    @_test_unrolling
    def _batches_slices(label_dim, tg, compare_fn):
        y_true = np.random.randint(0, 2, (3, 4, 8, label_dim))
        y_pred = np.random.uniform(0, 1, (3, 4, 8, label_dim))

        tg.set_cache(y_true, y_pred)
        tg.val_datagen.slices_per_batch = y_true.shape[1]
        return y_true, y_pred, tg, compare_fn

    def _make_traingen(metric_names, loss, label_dim):
        tg = TraingenDummy()
        tg.val_metrics = metric_names
        tg.loss = loss
        tg.set_shapes(batch_size=8, label_dim=label_dim)
        return tg

    def _make_compare_fn(metric_names):
        def compare_fn(m, yt, yp, test_name):
            for metric_name in metric_names:
                internal_score = m[metric_name]
                explicit_score = getattr(metric_fns, metric_name)(yt, yp)
                if isinstance(internal_score, (list, tuple)):
                    assert all(_is == _es for _is, _es in
                               zip(internal_score, explicit_score)), (
                                   test_name, metric_name,
                                   internal_score, explicit_score)
                assert (internal_score == explicit_score), (
                    test_name, metric_name, internal_score, explicit_score)
        return compare_fn

    def _test_binaries(test_fns):
        metric_names = ['binary_accuracy', 'tnr', 'tpr', 'tnr_tpr',
                        'f1_score', 'roc_auc_score']
        loss = 'binary_crossentropy'
        label_dim = 1
        compare_fn = _make_compare_fn(metric_names)

        for _test_fn in test_fns:
            tg = _make_traingen(metric_names, loss, label_dim)  # reset
            _test_fn(label_dim, tg, compare_fn)

    test_fns = (_batches, _unweighted_slices, _weighted_slices, _batches_slices)
    _test_binaries(test_fns)


@notify(tests_done)
def test_sklearn():
    y_true = np.random.randint(0, 2, (32,))
    y_pred = np.random.uniform(0, 1, (32,))
    assert (metric_fns.r2_score(y_true, y_pred) ==
            sklearn.metrics.r2_score(y_true, y_pred))


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
