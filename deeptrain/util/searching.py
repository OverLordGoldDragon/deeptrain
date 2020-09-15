"""Algorithms for searching combinations of various metric hyperparameters,
like classification prediction threshold, and best-performing subset of batches.

Useful for e.g. tuning a classifier's prediction threshold on validation data,
or tracking classifier calibration.
"""
# -*- coding: utf-8 -*-
from deeptrain import metrics
from .misc import argspec
import numpy as np


def find_best_predict_threshold(labels, preds, metric_fn, search_interval=.01,
                                search_min_max=(0, 1), max_is_best=True,
                                return_best_metric=False,
                                threshold_preference=0.5, verbosity=0):
    """Finds best scalar prediction threshold for an arbitrary metric function.

    Arguments:
        labels: np.ndarray
            Labels. All samples must be along dim0.
        preds: np.ndarray
            Predictions. All samples must be along dim0.
        metric_fn: function
            Metric function with input signature
            `(labels, preds, predict_threshold)`.
        search_interval: float
            Amount by which to increment predict_threshold from min to max
            of `search_min_max` in search of best threshold.
        search_min_max: tuple[float]
            Search bounds for best predict_threshold.
        max_is_best: bool
            "Best" means maximum if True, else minimum (metric).
        return_best_metric: bool
            If True, will also return metric_fn evaluated at the found best
            threshold. Default False.
        threshold_preference: float
            Select best metric yielding threshold that's closest to this, if
            there are multiple metrics that equal the best found.
        verbosity: int in (0, 1, 2)
            - 1: print found best predict threshold and metric metric.
            - 2: print a table of (threshold, metric, best metric) for every
              threshold computed.
            - 0: don't print anything.

    Returns:
        best_th: float
            Best prediction threshold
        best_metric: float
            Metric resulting from `best_th`; only returned if
            `return_best_metric` is True (default False).

    Finds best threshold by trying every threshold from min to max of
    `search_min_max`, incremented by `search_interval` (grid search).
    """
    _min = search_min_max[0] if search_min_max[0] != 0 else search_interval
    _max = search_min_max[1] if search_min_max[1] != 1 else .9999
    th_pref = threshold_preference

    th = _min
    best_th = th
    best_metric = 0
    if verbosity == 2:
        print("th", "metric", "best", sep="\t")

    while (th >= _min) and (th <= _max):
        metric = metric_fn(labels, preds, th)

        # find best that is closest to th_pref
        new_best = (metric > best_metric if max_is_best else
                    metric < best_metric)
        if new_best or (metric == best_metric and
                        abs(th - th_pref) < abs(best_th - th_pref)):
            best_th = round(th, 2)
            best_metric = metric

        if verbosity == 2:
            print("%.2f\t%.2f\t%.2f" % (th, metric, best_metric))
        th += search_interval

    if verbosity >= 1:
        print("Best predict th: %.2f w/ %.2f best metric" % (best_th, best_metric))

    return (best_th, best_metric) if return_best_metric else best_th


def find_best_subset(labels_all, preds_all, metric_fn, search_interval=.01,
                     search_min_max=(0, 1), max_is_best=True, subset_size=5):
    """Finds subset of batches yielding the best `metric_fn`.

    Arguments:
        labels_all: list[np.ndarray]
            Labels in `(batches, samples, *)` format, as arranged within
            :mod:`deeptrain.util.training`.
        preds_all: list[np.ndarray]
            Labels in `(batches, samples, *)` format, same as `labels_all`.
        metric_fn: function
            Metric function with input signature
            `(labels, preds, predict_threshold)`.
        search_interval: float
            Amount by which to increment predict_threshold from min to max
            of `search_min_max` in search of best threshold.
        search_min_max: tuple[float]
            Search bounds for best predict_threshold.
        max_is_best: bool
            "Best" means maximum if True, else minimum (metric).
        subset_size: int
            Size of best subset to find; must be `<= len(labels_all)` (but
            makes no sense if `==`).

    Returns:
        best_batch_idxs: list[int].
            Indices of best batches, w.r.t. original `labels_all` & `preds_all`.
        best_th: float
            Prediction threshold yielding best score on the found best subset.
        best_metric: float
            Metric computed the found best subset using `best_th`.

    Uses progressive elimination, and is *not* guaranteed to find the true
    best subset. Algorithm:

        1. Feed all labels & preds to `metric_fn`
        2. Remove "best" scoring batch from list of labels & preds
        3. Repeat 1-2 `subset_size` number of times.
    """
    def _flat(*lists):
        return [np.asarray(_list).ravel() for _list in lists]

    def _get_batch_metrics(labels_all, preds_all, metric_fn, th):
        batch_metrics = []
        for (batch_labels, batch_preds) in zip(labels_all, preds_all):
            bl_flat, bp_flat = _flat(batch_labels, batch_preds)
            if th is not None:
                batch_metrics += [metric_fn(bl_flat, bp_flat, th)]
            else:
                batch_metrics += [metric_fn(bl_flat, bp_flat)]
        return np.array(batch_metrics)

    def _best_batch_idx(labels_all, preds_all, metric_fn, th, max_is_best):
        best_batch_metrics = _get_batch_metrics(labels_all, preds_all,
                                                metric_fn, th)
        best_fn = np.max if max_is_best else np.min
        best_batch_idx = np.where(best_batch_metrics ==
                                  best_fn(best_batch_metrics))[0][0]
        return best_batch_idx

    def _best_batch_idx_thresholded(labels_all, preds_all, metric_fn,
                                    search_interval, search_min_max,
                                    max_is_best):
        def _compute_th_metrics(labels_all, preds_all, metric_multi_th_fn,
                                search_interval, search_min_max):
            mn, mx = search_min_max
            th_all = np.linspace(mn, mx, round((mx - mn) / search_interval) + 1)
            return th_all, metric_multi_th_fn(
                *_flat(labels_all, preds_all), th_all)

        metric_fn_name = metric_fn.__name__.split('.')[-1]
        metric_multi_th_fn = getattr(metrics, metric_fn_name + '_multi_th')
        th_all, th_metrics = _compute_th_metrics(
            labels_all, preds_all, metric_multi_th_fn,
            search_interval, search_min_max)

        best_th_metric_idx = list(th_metrics).index(max(th_metrics) if max_is_best
                                                    else min(th_metrics))
        best_th = th_all[best_th_metric_idx]

        return _best_batch_idx(labels_all, preds_all, metric_fn, best_th,
                               max_is_best)

    labels_all, preds_all = list(labels_all), list(preds_all)
    batch_idxs = list(range(len(preds_all)))
    best_labels, best_preds, best_batch_idxs = [], [], []

    while len(best_batch_idxs) < subset_size:
        if 'pred_threshold' in argspec(metric_fn):
            best_batch_idx = _best_batch_idx_thresholded(
                labels_all, preds_all, metric_fn,
                search_interval, search_min_max, max_is_best)
        else:
            best_batch_idx = _best_batch_idx(labels_all, preds_all, metric_fn,
                                             th=None, max_is_best=max_is_best)

        best_labels     += [labels_all.pop(best_batch_idx)]
        best_preds      += [preds_all.pop(best_batch_idx)]
        best_batch_idxs += [batch_idxs.pop(best_batch_idx)]

    best_labels_flat, best_preds_flat = _flat(best_labels, best_preds)

    if search_min_max is not None:  # 'pred_threshold' not in argspec(metric_fn)
        best_th, best_metric = find_best_predict_threshold(
            best_labels_flat, best_preds_flat, metric_fn, search_interval,
            search_min_max, return_best_metric=True)
    else:
        best_th = None
        best_metric = metric_fn(best_labels_flat, best_preds_flat)
    return best_batch_idxs, best_th, best_metric


def find_best_subset_from_history(metric, subset_size=5, max_is_best=True):
    """Finds subset of batches yielding the best metric, given pre-computed
    metrics. Simply orders metrics best-to-worst, and returns top `subset_size`
    of them. Exact.

    Arguments:
        metric: list[float]
            List of pre-computed metrics, arranged as `(batches, slices)`, or
            `(batches,)`; if former, will collapse slices as a mean.
        max_is_best: bool
            "Best" means maximum if True, else minimum (metric).

    Returns:
        list[int]: indices of best batches.
    """
    def _find_best(metric, subset_size):
        indices = list(range(len(metric)))
        idx_metric_pairs = [[i, m] for i, m in zip(indices, metric)]
        # sort by metrics, return indices
        #
        idx_metric_pairs.sort(key=lambda x: x[1], reverse=bool(max_is_best))
        return [idx_metric_pairs[j][0] for j in range(subset_size)]

    metric = np.asarray(metric)
    if metric.ndim > 1:  # (batches, slices)
        # collapse slices as means, since searching per batch not per slice
        metric = metric.mean(axis=1)

    return _find_best(metric, subset_size)
