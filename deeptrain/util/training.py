# -*- coding: utf-8 -*-
import numpy as np

from see_rnn import weight_loss
from deeptrain.backend.model_utils import model_loss_name
from .searching import find_best_predict_threshold, find_best_subset
from .searching import find_best_subset_from_history
from ._backend import NOTE, WARN
from .misc import argspec
from .. import metrics as metric_fns


def _update_temp_history(self, metrics, val=False):
    """Updates temporary history given `metrics`. If using batch-slices, ensures
    entries are grouped appropriately.

        - Gets train (`val=False`) or val (`=True`) metrics, and validates that
          `len(metrics)` of returned list matches len of `train_metrics` or
          `val_metrics`.
        - Validates that each element of `metrics` is a numeric (e.g. float)
    """
    def _get_metric_names(metrics, val):
        metric_names = self.val_metrics if val else self.train_metrics
        if (not val or (val and 'evaluate' in self._eval_fn_name)
            ) and len(metric_names) != len(metrics):
            raise Exception(f"{len(metric_names)} != {len(metrics)}, \n"
                            f"{metric_names}\n{metrics}")
        return metric_names

    def _get_temp_history(val):
        def _validate_temp_history(temp_history):
            for name, value in temp_history.items():
                if not isinstance(value, list):
                    print(NOTE, "`temp_history` is non-list; attempting casting")
                    temp_history[name] = list(value)
            return temp_history

        temp_history = self.val_temp_history if val else self.temp_history
        temp_history = _validate_temp_history(temp_history)
        return temp_history

    def _get_slice_info(val):
        datagen = self.val_datagen if val else self.datagen
        slice_idx = getattr(datagen, 'slice_idx', None)
        no_slices = slice_idx is None
        slices_per_batch = getattr(datagen, 'slices_per_batch', None)
        return no_slices, slice_idx, slices_per_batch

    def _try_append_with_fix(temp_history):
        try:
            temp_history[name][-1].append(value)
        except:
            # fallback in case TrainGenerator was saved/loaded from an incremented
            # slice_idx and temp_history was empty, which will fail the `try`
            print(WARN, "unable to append to `temp_history`; OK if right "
                  "after load() -- attempting fix via append()...")
            temp_history[name].append(value)

    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]
    metric_names = _get_metric_names(metrics, val)
    temp_history = _get_temp_history(val)
    no_slices, slice_idx, slices_per_batch = _get_slice_info(val)

    for name, value in zip(metric_names, metrics):
        if (np.ndim(value) != 0 or isinstance(value, dict)):
            raise TypeError(f"got non-scalar value ({type(value)}) for metric "
                            f"{name}")

        if no_slices or slice_idx == 0:
            temp_history[name].append([])
        _try_append_with_fix(temp_history)

        if not no_slices and slice_idx == (slices_per_batch - 1):
            temp_history[name][-1] = np.mean(temp_history[name][-1])


def get_sample_weight(self, class_labels, val=False, slice_idx=None,
                      force_unweighted=False):
    """Make `sample_weight` to feed to `model` based on `class_labels`
    (and, if applicable, weighted slices).

    Arguments:
        class_labels: np.ndarray
            Classification class labels to map sample weights according to
            `class_weights`.

            >>> class_weights = {0: 4, 1: 1, 2: 2.5}
            >>> class_labels  == [1, 1, 0, 1, 2]    # if
            >>> sample_weight == [4, 4, 1, 4, 2.5]  # then

        val: bool
            Whether to use `class_weights` (False) or `val_class_weights` (True).
            If using sliced weights, whether to use `slices_per_batch` of
            `datagen` (False) or `val_datagen` (False).
            Further, if True, will get weighted `sample_weight` if either of
            `loss_weighted_slices_range` or `pred_weighted_slices_range`
            are set (False requires former to be set).
        slice_idx: int / None
            Index of slice to get `sample_weight` for, if using slices.
            If None, will return all `slices_per_batch` number of `sample_weight`.
        force_unweighted: bool
            Get slice-unweighted `sample_weight` regardless of slice usage;
            used internally by :meth:`._get_weighted_sample_weight` to
            break recursion.
    """
    def _get_unweighted(class_labels, val):
        class_labels = _unroll_into_samples(len(self.model.output_shape),
                                            class_labels)
        cw = self.val_class_weights if val else self.class_weights

        if cw is not None:
            if sum(cw.keys()) > 1 and class_labels.ndim == 2:
                class_labels = class_labels.argmax(axis=1)  # one-hot to dense
            return np.asarray([cw[int(l)] for l in class_labels])
        return np.ones(class_labels.shape[0])

    def _get_weighted(class_labels, val, slice_idx):
        return _get_weighted_sample_weight(
            self, class_labels, val, self.loss_weighted_slices_range, slice_idx)

    loss_weighted = bool(self.loss_weighted_slices_range)
    pred_weighted = bool(self.pred_weighted_slices_range)
    either_weighted = loss_weighted or pred_weighted
    get_weighted = loss_weighted or (val and either_weighted)

    if force_unweighted or not get_weighted:
        return _get_unweighted(class_labels, val)
    else:
        return _get_weighted(class_labels, val, slice_idx)


def _get_weighted_sample_weight(self, class_labels_all, val=False,
                                weight_range=(0.5, 1.5), slice_idx=None):
    """Gets `slices_per_batch` number of `sample_weight`, scaled linearly
    from min to max of `weight_range`, over `slices_per_batch` number of steps.

    >>> weight_range == (0.5, 1.5)
    >>> class_weights == {0: 1, 1: 5}  # val = False
    >>> slices_per_batch == 3
    >>> slice_idx == None  # get all
    >>> class_labels_all == [[0, 0, 1], [0, 0, 1], [0, 1, 1]]
    ...
    >>> [[0.5, 0.5, 2.5],
    ...  [1.0, 1.0, 5.0],
    ...  [1.5, 7.5, 7.5]]
    """
    def _sliced_sample_weight(class_labels_all, slice_idx, val):
        sw_all = []
        for batch_labels in class_labels_all:
            sw_all.append([])
            for slice_labels in batch_labels:
                sw = get_sample_weight(self, slice_labels, val, slice_idx,
                                       force_unweighted=True)  # break recursion
                sw_all[-1].append(sw)
        sw_all = np.asarray(sw_all)
        if sw_all.ndim >= 3 and sw_all.shape[0] == 1:
            sw_all = sw_all.squeeze(axis=0)
        return sw_all

    # `None` as in not passed in, not datagen-absent
    validate_n_slices = slice_idx is None
    class_labels_all = self._validate_class_data_shapes(
        {'class_labels_all': class_labels_all}, val, validate_n_slices)

    n_slices = (self.val_datagen if val else self.datagen).slices_per_batch

    sw = _sliced_sample_weight(class_labels_all, slice_idx, val)
    sw = self._validate_class_data_shapes({'sample_weight_all': sw},
                                          val, validate_n_slices)
    sw_weights = np.linspace(*weight_range, n_slices
                             ).reshape([1, n_slices] + [1] * (sw.ndim - 2))

    sw = sw * sw_weights
    if slice_idx is not None:
        sw = sw[:, slice_idx]
    return sw.squeeze()


def _set_predict_threshold(self, predict_threshold, for_current_iter=False):
    """Set `predict_threshold` and maybe `dynamic_predict_threshold`."""
    if not for_current_iter and self.dynamic_predict_threshold is not None:
        self.dynamic_predict_threshold = predict_threshold
    self.predict_threshold = predict_threshold


def _get_val_history(self, for_current_iter=False):
    """Compute validation metrics from history ('evaluate'-mode), or
    cache ('predict'-mode).

    `for_current_iter` is True when inside of :meth:`.validate` loop,
    fitting individual batches/slices, and is False :meth:`_on_val_end`,
    where metrics over the entire validation dataset are computed.
    In latter, best subset is found (if applicable).
    """
    if self.best_subset_size and not for_current_iter:
        return _get_best_subset_val_history(self)

    if 'evaluate' in self._eval_fn_name:
        return {metric: np.mean(values) for metric, values in
                self.val_temp_history.items()}

    def _find_and_set_predict_threshold():
        if (self.dynamic_predict_threshold_min_max is None or
            self.dynamic_predict_threshold is None):
            search_min_max = (self.predict_threshold, self.predict_threshold)
        else:
            search_min_max = self.dynamic_predict_threshold_min_max

        pred_threshold = find_best_predict_threshold(
            labels_all_norm, preds_all_norm, self.key_metric_fn,
            search_interval=.01,
            search_min_max=search_min_max)
        self._set_predict_threshold(pred_threshold, for_current_iter)

    def _unpack_and_transform_data(for_current_iter):
        if for_current_iter:
            labels_all = self._labels_cache[-1].copy()
            preds_all  = self._preds_cache[-1].copy()
            sample_weight_all = self._sw_cache[-1].copy()
            class_labels_all = self._class_labels_cache[-1].copy()
        else:
            labels_all = self._labels_cache.copy()
            preds_all  = self._preds_cache.copy()
            sample_weight_all = self._sw_cache.copy()
            class_labels_all = self._class_labels_cache.copy()
        return self._transform_eval_data(labels_all, preds_all,
                                         sample_weight_all, class_labels_all,
                                         return_as_dict=False)

    # `class_labels_all` currently unused (used only for getting sample_weight);
    # may be useful in the future
    (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all
     ) = _unpack_and_transform_data(for_current_iter)

    if self.dynamic_predict_threshold is not None:
        _find_and_set_predict_threshold()

    return self._compute_metrics(labels_all_norm, preds_all_norm,
                                 sample_weight_all)


def _get_best_subset_val_history(self):
    """Returns history entry for best `best_subset_size` number of validation
    batches, and sets `best_subset_nums`.

    Ex: given 10 batches, a "best subset" of 5 is the set of 5 batches that
    yields the best (highest/lowest depending on `max_is_best`) `key_metric`.
    Useful for model ensembling in specializing member models on different
    parts of data.
    """
    def _unpack_and_transform_data():
        labels_all = self._labels_cache.copy()
        preds_all  = self._preds_cache.copy()
        sample_weight_all = self._sw_cache.copy()
        class_labels_all = self._class_labels_cache.copy()
        return self._transform_eval_data(labels_all, preds_all,
                                         sample_weight_all, class_labels_all,
                                         unroll_into_samples=False)

    def _find_best_subset_from_preds(d):
        def _merge_slices_samples(*arrs):
            ls = []
            for x in arrs:
                ls.append(x.reshape(x.shape[0], x.shape[1] * x.shape[2],
                                    *x.shape[3:]))
            return ls

        if 'pred_threshold' not in argspec(self.key_metric_fn):
            search_min_max = None
        elif (self.dynamic_predict_threshold_min_max is None or
              self.dynamic_predict_threshold is None):
            search_min_max = (self.predict_threshold, self.predict_threshold)
        else:
            search_min_max = self.dynamic_predict_threshold_min_max

        la_norm, pa_norm = _merge_slices_samples(d['labels_all_norm'],
                                                 d['preds_all_norm'])
        best_subset_idxs, pred_threshold, _ = find_best_subset(
            la_norm, pa_norm,
            search_interval=.01,
            search_min_max=search_min_max,
            metric_fn=self.key_metric_fn,
            subset_size=self.best_subset_size)
        return best_subset_idxs, pred_threshold

    def _find_best_subset_from_history():
        metric = self.val_temp_history[self.key_metric]
        best_subset_idxs = find_best_subset_from_history(
            metric, self.best_subset_size, self.max_is_best)
        return best_subset_idxs

    def _best_subset_metrics_from_history(best_subset_idxs):
        return {name: np.asarray(metric)[best_subset_idxs].mean()
                for name, metric in self.val_temp_history.items()}

    def _best_subset_metrics_from_preds(d, best_subset_idxs):
        def _filter_by_indices(indices, *arrs):
            return [np.asarray([x[idx] for idx in indices]) for x in arrs]

        if not best_subset_idxs:
            raise Exception("`best_subset_idxs` is empty")
        ALL = _filter_by_indices(best_subset_idxs, d['labels_all_norm'],
                                 d['preds_all_norm'], d['sample_weight_all'])

        (labels_all_norm, preds_all_norm, sample_weight_all
         ) = _unroll_into_samples(len(self.model.output_shape), *ALL)

        return self._compute_metrics(labels_all_norm, preds_all_norm,
                                     sample_weight_all)

    if 'evaluate' in self._eval_fn_name:
        best_subset_idxs = _find_best_subset_from_history()
    elif 'predict' in self._eval_fn_name:
        d = _unpack_and_transform_data()
        best_subset_idxs, pred_threshold = _find_best_subset_from_preds(d)
        if self.dynamic_predict_threshold is not None:
            self._set_predict_threshold(pred_threshold)
    else:
        raise ValueError("unknown `eval_fn_name`: %s" % self._eval_fn_name)

    self.best_subset_nums = np.array(self._val_set_name_cache)[best_subset_idxs]
    if 'evaluate' in self._eval_fn_name:
        return _best_subset_metrics_from_history(best_subset_idxs)
    else:
        return _best_subset_metrics_from_preds(d, best_subset_idxs)


def _get_api_metric_name(name, loss_name, alias_to_metric_name_fn=None):
    if name == 'loss':
        api_name = loss_name
    elif name in ('accuracy', 'acc'):
        if loss_name == 'categorical_crossentropy':
            api_name = 'categorical_accuracy'
        elif loss_name == 'sparse_categorical_crossentropy':
            api_name = 'sparse_categorical_accuracy'
        else:
            api_name = 'binary_accuracy'
    else:
        api_name = name
    if alias_to_metric_name_fn is not None:
        api_name = alias_to_metric_name_fn(api_name)
    return api_name


def _compute_metric(self, data, metric_name=None, metric_fn=None):
    """Compute metric given labels, preds, and sample weights or prediction
    threshold where applicable - and metric name or function.
    """
    def _del_if_not_in_metric_fn(name, data, metric_fn):
        if name in data and name not in argspec(metric_fn):
            del data[name]

    if metric_name is not None:
        metric_name = self._alias_to_metric_name(metric_name)
        if metric_name in self.custom_metrics:
            metric_fn = self.custom_metrics[metric_name]
        else:
            metric_fn = getattr(metric_fns, metric_name)

    _del_if_not_in_metric_fn('pred_threshold', data, metric_fn)
    _del_if_not_in_metric_fn('sample_weight', data, metric_fn)
    return metric_fn(**data)


def _compute_metrics(self, labels_all_norm, preds_all_norm, sample_weight_all):
    """Computes metrics from labels, predictions, and sample weights, via
    :meth:`_compute_metric`.

    Iterates over metric names in `val_metrics`:

        - `name == 'loss'`: fetches loss name from `model.loss`, then function
          from `deeptrain.util.metrics`, computes metric, and adds model
          weight penalty loss (L1/L2). Loss from `model.evaluate()` may still
          differ, as no other regularizer loss is accounted for.
        - `name == key_metric`: computes metric with `key_metric_fn`.
        - `name in custom_metrics`: computes metric with `custom_metrics[name]`
          function.
        - `name` none of the above: passes name and `model.loss` to
          :meth:`_compute_metric`.

    Ensures computed metrics are scalars (numbers, instead of lists, tuples, etc).
    """
    def _ensure_scalar_metrics(metrics):
        def _ensure_is_scalar(metric):
            if np.ndim(metric) != 0:
                if (metric.ndim > 1):
                    raise Exception("unfamiliar metric.ndim: %s" % metric.ndim
                                    + "; expected <= 1.")
                metric = metric.mean()
            return metric

        for name, metric in metrics.items():
            if isinstance(metric, list):
                for i, m in enumerate(metric):
                    metrics[name][i] = _ensure_is_scalar(m)
            else:
                metrics[name] = _ensure_is_scalar(metric)
        return metrics

    metric_names = self.val_metrics.copy()
    metrics = {}
    for name in metric_names:
        data = dict(y_true=labels_all_norm,
                    y_pred=preds_all_norm,
                    sample_weight=sample_weight_all,
                    pred_threshold=self.predict_threshold)

        if name == 'loss':
            api_name = _get_api_metric_name('loss', model_loss_name(self.model),
                                            self._alias_to_metric_name)
            metrics[name] = self._compute_metric(data, metric_name=api_name)
            metrics[name] += weight_loss(self.model)
        elif name == self.key_metric:
            metrics[name] = self._compute_metric(data,
                                                 metric_fn=self.key_metric_fn)
        else:
            api_name = _get_api_metric_name(name, model_loss_name(self.model),
                                            self._alias_to_metric_name)
            metrics[name] = self._compute_metric(data, metric_name=api_name)

    metrics = _ensure_scalar_metrics(metrics)
    return metrics


def _unroll_into_samples(out_ndim, *arrs):
    """Flatten samples, slices, and batches dims into one:
        - `(batches, slices, samples, *output_shape)` ->
          `(batches * slices * samples, *output_shape)`
        - `(batches, samples, *output_shape)` ->
          `(batches * samples, *output_shape)`

    `*arrs` are standardized (fed after :meth:`_transform_eval_data`),
    so the minimal case is `(1, 1, *output_shape)`, which still correctly
    reshapes into `(1, *output_shape)`. Cases:

    >>> #       (32, 1) -> (32, 1)
    ... #    (1, 32, 1) -> (32, 1)
    ... # (1, 1, 32, 1) -> (32, 1)
    ... # (1, 3, 32, 1) -> (66, 1)
    ... # (2, 3, 32, 1) -> (122, 1)
    """
    ls = []
    for x in arrs:
        # unroll along non-out (except samples) dims
        x = x.reshape(-1, *x.shape[-(out_ndim - 1):])
        while x.shape[0] == 1:  # collapse non-sample dims
            x = x.squeeze(axis=0)
        ls.append(x)
    return ls if len(ls) > 1 else ls[0]


def _transform_eval_data(self, labels_all, preds_all, sample_weight_all,
                         class_labels_all, return_as_dict=True,
                         unroll_into_samples=True):
    """Prepare data for feeding to metrics computing methods.

        - Stanardize labels and preds shapes to the expected
          `(batches, *model.output_shape)`, or
          `(batches, slices, *model.output_shape)` if slices are used.
          See :meth:`_validate_data_shapes`.
        - Standardize `sample_weight` and `class_labels` shapes.
          See :meth:`_validate_class_data_shapes`.
        - Unroll data into samples (merge batches, slices, and samples dims).
          See :func:`_unroll_into_samples`.
    """
    def _transform_labels_and_preds(labels_all, preds_all, sample_weight_all,
                                    class_labels_all):
        d = self._validate_data_shapes({'labels_all': labels_all,
                                        'preds_all': preds_all})
        labels_all, preds_all = d['labels_all'], d['preds_all']

        # if `loss_weighted_slices_range` but not `pred_weighted_slices_range`,
        # will apply weighted sample weights on non weight-normalized preds
        if self.pred_weighted_slices_range is not None:
            # shapes: (batches, slices, samples, *) -> (batches, samples, *)
            preds_all_norm = self._weighted_normalize_preds(preds_all)
            labels_all_norm = labels_all[:, 0]  # collapse slices dim

            pmin, pmax = preds_all_norm.min(), preds_all_norm.max()
            if not (pmin >= 0 and pmax <= 1):
                raise ValueError("preds must lie within [0, 1], got "
                                 f"min={pmin}, max={pmax}")
        else:
            preds_all_norm = preds_all
            labels_all_norm = labels_all

        d = self._validate_class_data_shapes(
            {'sample_weight_all': sample_weight_all,
             'class_labels_all': class_labels_all})
        if self.pred_weighted_slices_range is not None:
            # can no longer use slice-weighted sample weights, so just take mean
            d['sample_weight_all'] = d['sample_weight_all'].mean(axis=1)
            d['class_labels_all'] = d['class_labels_all'][:, 0]

        return (labels_all_norm, preds_all_norm, d['sample_weight_all'],
                d['class_labels_all'])

    (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all,
     ) = _transform_labels_and_preds(labels_all, preds_all, sample_weight_all,
                                     class_labels_all)

    data = (labels_all_norm, preds_all_norm, sample_weight_all, class_labels_all)
    if unroll_into_samples:
        data = _unroll_into_samples(len(self.model.output_shape), *data)

    if return_as_dict:
        names = ('labels_all_norm', 'preds_all_norm', 'sample_weight_all',
                 'class_labels_all')
        return {name: x for name, x in zip(names, data)}
    else:
        return data


def _weighted_normalize_preds(self, preds_all):
    """Given batch-slices predictions, "weighs" binary (sigmoid) class predictions
    linearly according to `pred_weighted_slices_range`, in `slices_per_batch`
    steps. In effect, 0-class predictions from later slices are weighted
    greater than those from earlier, and likewise for 1-class.

    Norm logic: "0-class" is defined as predicting <0.5. 1 is subtracted
    from all predictions, so that 0-class preds are negative, and 1-class are
    positive. Preds are then *scaled* according to slice weights, so that
    greater weights correspond to more negative and more positive values. Preds
    are then shifted back to original [0, 1]. More negative values in shifted
    domain thus correspond to values closer to 0 in original domain, in a manner
    that weighs 0-class preds and 1-class preds equally.
    """
    def _validate_data(preds_all, n_slices):
        spb = self.val_datagen.slices_per_batch
        if n_slices != spb:
            raise Exception("`n_slices` inferred from `preds_all` differs"
                            " from `val_datagen.slices_per_batch` "
                            "(%s != %s)" % (n_slices, spb))
        pmin, pmax = preds_all.min(), preds_all.max()
        if not (pmin >= 0 and pmax <= 1):
            raise ValueError("preds must lie within [0, 1], got "
                             f"min={pmin}, max={pmax}")

    n_slices = preds_all.shape[1]
    # validate even if n_slices == 1 to ensure expected behavior
    # in metrics computation
    _validate_data(preds_all, n_slices)

    if n_slices == 1:
        return preds_all
    slice_weights = np.linspace(*self.pred_weighted_slices_range, n_slices)

    weighted_preds = []
    for preds in preds_all:
        weighted_preds.append([])
        for slice_idx, pred in enumerate(preds):
            additive_pred = pred - .5
            weighted_preds[-1] += [additive_pred * slice_weights[slice_idx]]

    weight_norm = np.sum(slice_weights)
    preds_norm = np.sum(np.array(weighted_preds), axis=1) / weight_norm + .5

    return preds_norm


def _validate_data_shapes(self, data, val=True,
                          validate_n_slices=True,
                          validate_last_dims_match_outs_shape=True,
                          validate_equal_shapes=True):
    """Ensures `data` entires are shaped `(batches, *model.output_shape)`,
    or `(batches, slices, *model.output_shape)` if using slices.

        - Validate `batch_size`, and that it's common to every batch/slice.
        - Validate `slices_per_batch`, and that it's common to every batch/slice,
          `if validate_n_slices`.

    Arguments:
        data: dict[str: np.ndarray]
            `{'labels_all': labels_all, 'preds_all': preds_all}`. Passed as
            self-naming dict to improve code readability in exception handling.
        val: bool
            Only relevant with `validate_n_slices==True`; if True, gets
            `slices_per_batch` from `val_datagen` - else, from `datagen`.
        validate_n_slices: bool
            (Default True) is set False when `slice_idx` is not None in
            in `_get_weighted_sample_weight`, which occurs during
            :meth:`.validate` when processing individual batch-slices.
            `slice_idx` is None :meth:`._on_val_end`.
        validate_last_dims_match_outs_shape: bool
            See :meth:`_validate_class_data_shapes`.
        validate_equal_shapes: bool
            See :meth:`_validate_class_data_shapes`.
    """
    def _validate_batch_size(data, outs_shape):
        batch_size = outs_shape[0]
        if batch_size is None:
            batch_size = self.batch_size or self._inferred_batch_size
            if batch_size is None:
                raise ValueError("unable to infer `batch_size`")

        for name, x in data.items():
            if batch_size not in x.shape:
                raise Exception(f"`{name}.shape` must include batch_size "
                                f"(={batch_size}) {x.shape}")
        return batch_size

    def _validate_iter_ndim(data, slices_per_batch, ndim):
        if slices_per_batch is not None:
            expected_iter_ndim = ndim + 2  # +(batches, slices)
        else:
            expected_iter_ndim = ndim + 1  # +(batches,)

        for name in data:
            dndim = data[name].ndim
            if dndim > expected_iter_ndim:
                raise Exception(f"{name}.ndim exceeds `expected_iter_ndim` "
                                f"({dndim} > {expected_iter_ndim}) "
                                f"-- {data[name].shape}")
            while data[name].ndim < expected_iter_ndim:
                data[name] = np.expand_dims(data[name], 0)
        return data

    def _validate_last_dims_match_outs_shape(data, outs_shape, ndim):
        for name, x in data.items():
            if x.shape[-ndim:] != outs_shape:
                raise Exception(f"last dims of `{name}` must equal "
                                f"model.output_shape [%s != %s]" % (
                                    x.shape[-ndim:], outs_shape))

    def _validate_equal_shapes(data):
        x = list(data.values())[0]
        if not all(y.shape == x.shape for y in data.values()):
            raise Exception("got unequal shapes for", ", ".join(list(data)))

    def _validate_n_slices(data, slices_per_batch):
        if slices_per_batch is not None:
            for name, x in data.items():
                if slices_per_batch not in x.shape:
                    raise Exception(f"{name} -- {x.shape}, {slices_per_batch} "
                                    "slices_per_batch")

    for name in data:
        data[name] = np.asarray(data[name])

    outs_shape = list(self.model.output_shape)
    outs_shape[0] = _validate_batch_size(data, outs_shape)
    outs_shape = tuple(outs_shape)

    ndim = len(outs_shape)
    slices_per_batch = getattr(self.val_datagen if val else self.datagen,
                               'slices_per_batch', None)
    data = _validate_iter_ndim(data, slices_per_batch, ndim)

    if validate_last_dims_match_outs_shape:
        _validate_last_dims_match_outs_shape(data, outs_shape, ndim)
    if validate_equal_shapes:
        _validate_equal_shapes(data)
    if validate_n_slices:
        _validate_n_slices(data, slices_per_batch)

    return data if len(data) > 1 else list(data.values())[0]


def _validate_class_data_shapes(self, data, val=True, validate_n_slices=False):
    """Standardize `sample_weight` and `class_labels` data. Same as
    :meth:`_validate_data_shapes`, except skips two validations:

        - `_validate_last_dims_match_outs_shape`; for `class_labels`, model
          output shapes can be same as input shapes as with autoencoders,
          but inputs can still have class labels, subjecting to `sample_weight`.
          `sample_weight` often won't share model output shape, as with e.g.
          multiclass classification, where individual classes aren't weighted.
        - `_equal_shapes`; per above, `data` entries may not have equal shapes.
    """
    return self._validate_data_shapes(data, val,
                                      validate_n_slices,
                                      validate_last_dims_match_outs_shape=False,
                                      validate_equal_shapes=False)
