# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from termcolor import cprint
from see_rnn import get_gradients, features_hist, detect_nans
from see_rnn import get_layer
from see_rnn.inspect_gen import _make_grads_fn, _get_grads_eager
from see_rnn.utils import _get_params
from .util._backend import K, WARN, TF_KERAS
from . import scalefig


def compute_gradient_norm(self, input_data, labels, sample_weight=None,
                          learning_phase=0, _id='*', mode='weights',
                          norm_fn=(np.sqrt, np.square), scope='local'):
    """Computes gradients w.r.t. layer weights or outputs per `_id`, and returns
    norm according to `norm_fn` and `scope`.

    Arguments:
        input_data: np.ndarray / list[np.ndarray] / supported formats
            Data w.r.t. which loss is to be computed for the gradient.
            List of arrays for multi-input networks. "Supported formats"
            is any valid input to `model`.
        labels: np.ndarray / list[np.ndarray] / supported formats
            Labels w.r.t. which loss is to be computed for the gradient.
        sample_weight: np.ndarray / list[np.ndarray] / supported formats
            kwarg to `model.fit()`, etc., weighting individual sample losses.
        learning_phase: bool / int[bool]
            - 1: use model in train mode
            - 0: use model in inference mode
        _id: str / int / list[str/int].
            - int -> idx; str -> name
            - idx: int. Index of layer to fetch, via model.layers[idx].
            - name: str. Name of layer (full or substring) to be fetched.
              Returns earliest match if multiple found.
            - list[str/int] -> treat each str element as name, int as idx.
              Ex: `['gru', 2]` gets (e.g.) weights of first layer with name
              substring 'gru', then of layer w/ idx 2.
            - `'*'` (wildcard) -> get (e.g.) outputs of all layers (except input)
              with 'output' attribute.
        mode: str in ('weights', 'outputs', 'gradients:weights',\
        'gradients:outputs')
            Whether to fetch layer weights, outputs, or gradients (w.r.t.
            outputs or weights).
        norm_fn: (function, function) / function
            Norm function(s) to apply to gradients arrays when gathering.
            `(np.sqrt, np.square)` for L2-norm, `np.abs` for L1-norm.
            Computed as: `outer_fn(sum(inner_fn(x) for x in data))`, where
            `outer_fn, inner_fn = norm_fn` if `norm_fn` is list/tuple, and
            `inner_fn = norm_fn` and `outer_fn = lambda x: x` otherwise.
        scope: str in ('local', 'global')
            Whether to apply `stat_fn` on individual gradient arrays, or sum of.

    Returns:
        Gradient norm(s). List of float if `scope == 'local'` (norms of weights),
        else float (`outer_fn(sum(sum(inner_fn(g)) for g in grads))`).

    TensorFlow optimizers do gradient clipping according to the `clipnorm` setting
    by comparing individual weights' L2-norms against `clipnorm`, and rescaling
    if exceeding. These L2 norms can be obtained using
    `norm_fn=(np.sqrt, np.square)` with `scope == 'local'` and `mode='weights'`.
    See:

        - `tensorflow.python.keras.optimizer_v2.optimizer_v2._clip_gradients`
        - `keras.optimizers.clip_norm`
        - `tensorflow.python.ops.clip_ops.clip_by_norm`
    """
    if scope not in ('local', 'global'):
        raise ValueError("`scope` must be one of: 'local', 'global' "
                         "(got '%s')" % scope)
    if isinstance(norm_fn, (tuple, list)):
        outer_fn, inner_fn = norm_fn
    else:
        outer_fn, inner_fn = lambda x: x, norm_fn

    sample_weight = _validate_sample_weight(self.model, sample_weight)
    grads = get_gradients(self.model, _id, input_data, labels, sample_weight,
                          learning_phase, mode=mode, as_dict=False)
    inner_sum = [np.sum(inner_fn(g)) for g in grads]
    if scope == 'local':
        # same as e.g. [np.sqrt(np.sum(np.square(g))) for g in grads], but faster
        return outer_fn(inner_sum)
    else:
        return outer_fn(np.sum(inner_sum))


def gradient_norm_over_dataset(self, val=False, learning_phase=0, mode='weights',
                               norm_fn=(np.sqrt, np.square), stat_fn=np.median,
                               n_iters=None, prog_freq=10, w=1, h=1):
    """Aggregates gradient norms over dataset, one iteration at a time. Useful
    for estimating value of gradient clipping, `clipnorm`, to use.
    Plots a histogram of gathered data when finished. Also see
    :meth:`compute_gradient_norm`.

    Arguments:
        val: bool
            - True:  gather over `val_datagen` batches
            - False: gather over `datagen` batches
        learning_phase: bool / int[bool]
            - True:  get gradients of model in train mode
            - False: get gradients of model in inference mode
        mode: str in ('weights', 'outputs')
            Whether to get gradients with respect to layer weights or outputs.
        norm_fn: (function, function) / function
            Norm function(s) to apply to gradients arrays when gathering.
            `(np.sqrt, np.square)` for L2-norm, `np.abs` for L1-norm.
            Computed as: `outer_fn(sum(inner_fn(g) for g in grads))`, where
            `outer_fn, inner_fn = norm_fn` if `norm_fn` is list/tuple, and
            `inner_fn = norm_fn` and `outer_fn = lambda x: x` otherwise.
        stat_fn: function
            Aggregate function to apply on computed norms. If `np.mean`,
            will gather mean of gradients; if `np.median`, the median, etc.
            Computed as: `stat_fn(outer_fn(sum(inner_fn(g) for g in grads)))`.
        n_iters: int / None
            Number of expected iterations over entire dataset. Can be used to
            iterate over subset of entire dataset. If None, will return upon
            `DataGenerator.all_data_exhausted`.
        prog_freq: int
            How often to print `f'|{batch_idx}'`, and `'.'` otherwise,
            in terms of number of batches (*not* iterations, but are same if not
            using slices). E.g. 5: `....|5....|10....|15`.
        w, h: float
            Scale figure width & height, respectively.

    Returns:
        grad_norms: np.ndarray
            Norms of gradients for every iteration.
            Shape: `(iters_processed, n_params)`, where `n_params` is number
            of gradient arrays whose norm stats were computed at each iteration.
        batches_processed: int
            Number of batches processed.
        iters_processed: int
            Number of iterations processed (if using e.g. 4 slices per batch,
            will equal `4 * batches_processed`).
    """
    def _init_notify(learning_phase, val):
        dg_name = 'val_datagen' if val else 'datagen'
        mode_name = "train" if learning_phase == 1 else "inference"
        print("Computing gradient l2-norm over", dg_name, "batches, in",
              mode_name, "mode")

    def _print_results(grad_norms, batches_processed, iters_processed, val):
        dg_name = 'val_datagen' if val else 'datagen'
        print(("\nGRADIENT L2-NORM (AVG, MAX) = ({:.3f}, {:.3f}), computed over "
        	   "{} batches, {} {} updates").format(
        		   grad_norms.mean(), grad_norms.max(), batches_processed,
        		   iters_processed, dg_name))

    def _plot(grad_norms, w=1, h=1):
        bins = min(600, len(grad_norms))
        plt.hist(grad_norms.ravel(), bins=bins)
        plt.gcf().set_size_inches(9 * w, 4 * h)
        scalefig(plt.gcf())
        plt.show()

    def _compute_gradient_norm_stat(model, x, y, sw):
        grads = grads_fn(x, y, sw)
        return stat_fn(outer_fn([np.sum(inner_fn(g)) for g in grads]))

    def gather_fn(data, model, x, y, sw):
        newdata = _compute_gradient_norm_stat(model, x, y, sw)
        data.append(newdata)
        return data

    _init_notify(learning_phase, val)
    if isinstance(norm_fn, (tuple, list)):
        outer_fn, inner_fn = norm_fn
    else:
        outer_fn, inner_fn = lambda x: x, norm_fn

    grads_fn = _make_gradients_fn(self.model, learning_phase, mode)

    grad_norms, batches_processed, iters_processed = _gather_over_dataset(
        self, gather_fn, val, n_iters, prog_freq)
    grad_norms = np.array(grad_norms)

    _print_results(grad_norms, batches_processed, iters_processed, val)
    _plot(grad_norms, w, h)

    return grad_norms, batches_processed, iters_processed


def gradient_sum_over_dataset(self, val=False, learning_phase=0, mode='weights',
                              n_iters=None, prog_freq=10, plot_kw={}):
    """Computes cumulative sum of gradients over dataset, one iteration at a time,
    preserving full array shapes. Useful for computing mean of gradients over
    dataset, or other aggregate metrics.

    Arguments:
        val: bool
            - True:  gather over `val_datagen` batches
            - False: gather over `datagen` batches
        learning_phase: bool / int[bool]
            - True:  get gradients of model in train mode
            - False: get gradients of model in inference mode
        mode: str in ('weights', 'outputs')
            Whether to get gradients with respect to layer weights or outputs.
        n_iters: int / None
            Number of expected iterations over entire dataset. Can be used to
            iterate over subset of entire dataset. If None, will return upon
            `DataGenerator.all_data_exhausted`.
        prog_freq: int
            How often to print `f'|{batch_idx}'`, and `'.'` otherwise,
            in terms of number of batches (*not* iterations, but are same if not
            using slices). E.g. 5: `....|5....|10....|15`.
        plot_kw: dict
            Kwargs to pass to `see_rnn.features_hist`; defaults to
            `{'share_xy': False, 'center_zero': True}`.

    Returns:
        grad_sum: dict[str: np.ndarray]
            Gradient arrays summed over dataset. Structure:
            `{name: array, name: array, ...}`, where `name` is name of
            weight array or layer output.
        batches_processed: int
            Number of batches processed.
        iters_processed: int
            Number of iterations processed (if using e.g. 4 slices per batch,
            will equal `4 * batches_processed`).
    """
    def _init_notify(learning_phase, val):
        dg_name = 'val_datagen' if val else 'datagen'
        mode_name = "train" if learning_phase == 1 else "inference"
        print("Computing gradients sum over", dg_name, "batches, in",
              mode_name, "mode")

    def _print_results(grads_sum, batches_processed, iters_processed, val):
        dg_name = 'val_datagen' if val else 'datagen'
        print(("\nGRADIENTS SUM computed over {} batches, {} {} updates:").format(
            batches_processed, iters_processed, dg_name))

    def _plot(grads_sum, plot_kw):
        defaults = {'share_xy': False, 'center_zero': True}
        for k, v in defaults.items():
            if k not in plot_kw:
                plot_kw[k] = v
        data = list(grads_sum.values())
        features_hist(data, annotations=list(grads_sum), **plot_kw)

    def gather_fn(data, model, x, y, sw):
        newdata = grads_fn(x, y, sw)

        if not data:
            return newdata
        for k, v in newdata.items():
            for i, x in enumerate(v):
                data[k][i] += x
        return data

    _init_notify(learning_phase, val)
    grads_fn = _make_gradients_fn(self.model, learning_phase, mode,
                                  return_names=True)

    grads_sum, batches_processed, iters_processed = _gather_over_dataset(
        self, gather_fn, val, n_iters, prog_freq)

    _print_results(grads_sum, batches_processed, iters_processed, val)
    _plot(grads_sum, plot_kw)

    return grads_sum, batches_processed, iters_processed


def _gather_over_dataset(self, gather_fn, val=False, n_iters=None, prog_freq=10):
    """Iterates over `DataGenerator`, applying `gather_fn` to every batch
    (or slice). Stops after `n_iters`, or when `DataGenerator.all_data_exhausted`
    if `n_iters is None`. Useful for monitoring quantities over the course of
    training or inference,.

    `gather_fn` recursively updates `data`; as such, it can be used to
    append to a list, update a dictionary, operate on an array, etc. Review
    source code for exact logic.
    """
    def _init_notify(val):
        print(WARN, "val_datagen" if val else "datagen", "states will be reset")
        print("'.' = slice processed, '|' = batch processed")

    def _print_progress(dg, batches_processed, prog_freq):
        if not dg.batch_exhausted:
            prog_mark = '.'
        else:
            prog_mark = '|'
            batches_processed += 1
            if batches_processed % max(1, prog_freq) == 0:
                prog_mark += str(batches_processed)
        print(end=prog_mark)
        return batches_processed

    def _gather(gather_fn, n_iters, val):
        def cond(iters_processed, n_iters, dg):
            if n_iters is None:
                return not dg.all_data_exhausted
            return iters_processed < n_iters

        dg.all_data_exhausted = False
        gathered = []
        batches_processed = 0
        iters_processed = 0

        while cond(iters_processed, n_iters, dg):
            # recursively update `gathered` according to `gather_fn`
            dg.advance_batch()
            x, y, sw = self.get_data(val=val)
            gathered = gather_fn(gathered, self.model, x, y, sw)
            dg.update_state()
            batches_processed = _print_progress(dg, batches_processed, prog_freq)
            iters_processed += 1
        return gathered, batches_processed, iters_processed

    dg = self.val_datagen if val else self.datagen
    _init_notify(val)

    dg.reset_state()
    (gathered, batches_processed, iters_processed
     ) = _gather(gather_fn, n_iters, val)
    dg.reset_state()

    return gathered, batches_processed, iters_processed


def _make_gradients_fn(model, learning_phase, mode, return_names=False):
    """Makes reusable gradient-getter function, separately for TF Eager & Graph
    execution. Eager variant is pseudo-reusable; gradient tensors are still
    fetched all over - graph should be significantly faster.
    """
    def _fn_graph(model, learning_phase, mode, params):
        # make grads_fn only once instead of repeatedly calling
        # `get_gradients` for potentially massive speedup due to
        # not rebuilding graph
        _g_fn = _make_grads_fn(model, params=params, mode=mode)

        def grads_fn(x, y, sw):
            ins = [x, y]
            if _sample_weight_built(model):
                if sw is None:
                    if isinstance(x, list):
                        sw = []
                        for data in x:
                            # extend to each input
                            sw.append(np.ones(len(data)))
                    else:
                        sw = [np.ones(len(x))]
                ins.append(sw)

            for i, data in enumerate(ins):
                if not isinstance(data, (list, tuple)):
                    ins[i] = [data]
            ins = [x for data in ins for x in data]  # flatten list

            return _g_fn([*ins, bool(learning_phase)])
        return grads_fn

    def _fn_eager(model, learning_phase, mode, params):
        return lambda x, y, sw: _get_grads_eager(model, x, y, sw,
                                                 learning_phase, params=params)

    _id = [l.name for l in model.layers]
    layers = get_layer(model, _id)
    params = _get_params(model, layers, mode=mode, verbose=0)

    if TF_KERAS and tf.executing_eagerly():
        _grads_fn = _fn_eager(model, learning_phase, mode, params)
    else:
        _grads_fn = _fn_graph(model, learning_phase, mode, params)

    if return_names:
        def grads_fn(x, y, sw):
            return {p.name: g for p, g in zip(params, _grads_fn(x, y, sw))}
    else:
        def grads_fn(x, y, sw):
            return _grads_fn(x, y, sw)
    return grads_fn


def print_dead_weights(model, dead_threshold=1e-7, notify_above_frac=1e-3,
                       notify_detected_only=False):
    """Print names of dead weights and their proportions. Useful for debugging
    vanishing and exploding gradients, or quantifying sparsity.

    Arguments:
        model: models.Model / models.Sequential (keras / tf.keras)
            The model.
        dead_threshold: float
            Threshold below which to count the weight as "dead", in
            absolute value.
        notify_above_frac: float
            Print only if fraction of weights counted "dead" exceeds this
            (e.g. if there are 11 absolute values < `dead_threshold` out of 1000).
        notify_detected_only: bool
            - True:  print text only if dead weights are discovered
            - False: print a "not found given thresholds" message when appropriate
    """
    def _print_dead(frac_dead, w_name, notify_above_frac):
        precision = int(np.ceil(-np.log10(notify_above_frac)))
        perc_dead = f'%.{precision}f' % (100 * frac_dead) + '%'

        cprint("{} dead -- '{}'".format(perc_dead, w_name), 'red')

    weight_names, weight_values = _get_weight_names_and_values(model)
    has_dead_worth_notifying = False
    has_dead = False

    for w_name, w_value in zip(weight_names, weight_values):
        num_dead = np.sum(np.abs(w_value) < dead_threshold)
        if num_dead > 0:
            has_dead = True

        frac_dead = num_dead / w_value.size
        if frac_dead > notify_above_frac:
            has_dead_worth_notifying = True
            _print_dead(frac_dead, w_name, notify_above_frac)

    if has_dead_worth_notifying:
        print("L = layer index, W = weight tensor index")
    elif not notify_detected_only:
        if has_dead:
            _txt = "Dead weights detected, but didn't notify; "
        else:
            _txt = "No dead weights detected in any trainable layers; "
        print(_txt + "(dead_threshold, notify_above_frac) = ({}, {})".format(
            dead_threshold, notify_above_frac))


def print_nan_weights(model, notify_detected_only=False):
    """Print names of NaN/Inf weights and their proportions. Useful for debugging
    exploding or buggy gradients.

    Arguments:
        model: models.Model / models.Sequential (keras / tf.keras)
            The model.
        notify_detected_only: bool
            - True:  print text only if dead weights are discovered
            - False: print a "none found" message if no NaNs were found
    """
    weight_names, weight_values = _get_weight_names_and_values(model)
    has_nan = False

    for w_name, w_value in zip(weight_names, weight_values):
        num_nan = np.sum(np.isnan(w_value) + np.isinf(w_value))
        txt = detect_nans(w_value, include_inf=True)
        if txt:
            if not has_nan:  # does have `if txt`, but `has_nan` not set yet
                print(flush=True)  # newline
            cprint("{} -- '{}'".format(txt, w_name), color='red', flush=True)
        if num_nan > 0:
            has_nan = True

    if has_nan:
        print("L = layer index, W = weight tensor index", end='')
    elif not notify_detected_only:
        print("No NaN weights detected in any trainable layers")


def print_large_weights(model, large_threshold=3, notify_above_frac=1e-3,
                        notify_detected_only=False):
    """Print names of weights in excess of set absolute value, and their
    proportions; excludes Inf. Useful for debugging exploding or buggy gradients.

    Arguments:
        model: models.Model / models.Sequential (keras / tf.keras)
            The model.
        large_threshold: float
            Threshold above which to count the weight's absolute value as "large".
        notify_above_frac: float
            Print only if fraction of weights counted "large" exceeds this
            (e.g. if there are 11 absolute values < `large_threshold`
             out of 1000).
        notify_detected_only: bool
            - True:  print text only if dead weights are discovered
            - False: print a "none found" message if no NaNs were found
    """
    def _get_txt(w_value, num_large):
        num_total = np.asarray(w_value).size
        perc = 100 * num_large / num_total
        txt = ''
        if perc > 0:
            if perc < .1:
                num = int((perc / 100) * num_total)  # show as quantity
                txt = "{:d}% Large".format(num)
            else:
                txt = "{:.1f}% Large".format(perc)  # show as percent
        return txt

    weight_names, weight_values = _get_weight_names_and_values(model)
    has_large = False

    for w_name, w_value in zip(weight_names, weight_values):
        num_nan = np.sum(np.isnan(w_value) + np.isinf(w_value))
        num_large = np.sum(np.abs(w_value) > large_threshold) - num_nan
        txt = _get_txt(w_value, num_large)
        if txt:
            if not has_large:  # does have `if txt`, but `has_large` not set yet
                print(flush=True)  # newline
            cprint("{} -- '{}'".format(txt, w_name), color='red', flush=True)
        if num_large > 0:
            has_large = True

    if has_large:
        print("L = layer index, W = weight tensor index", end='')
    elif not notify_detected_only:
        print("No NaN weights detected in any trainable layers")


def _get_weight_names_and_values(model):
    weight_names   = [w.name for layer in model.layers for w in layer.weights]
    weight_tensors = [w      for layer in model.layers for w in layer.weights]
    weight_values  = K.batch_get_value(weight_tensors)
    return weight_names, weight_values


def _sample_weight_built(model):
    """In Graph execution, `model._feed_sample_weights` isn't built unless
    model is compiled with `sample_weight_mode` set, or `train_on_batch` or
    `test_on_batch` is called w/ `sample_weight` passed.
    """
    return (not hasattr(model, '_feed_sample_weights')  # Eager case
            or (model._feed_sample_weights is not None and
                len(model._feed_sample_weights) > 0))


def _validate_sample_weight(model, sample_weight):
    if sample_weight is None or len(sample_weight) == 0:
        return None
    if not _sample_weight_built(model):
        print(WARN, "passed `sample_weight` but model doesn't have it built; "
              "compile with `sample_weight_mode='samplewise'` or call "
              "`train_on_batch` w/ `sample_weight`")
        return None
    return sample_weight


def interrupt_status(self) -> (bool, bool):
    """Prints whether `TrainGenerator` was interrupted (e.g. `KeyboardInterrupt`,
    or via exception) during :meth:`train` and :meth:`validate`. Returns
    bools (True for interrupted, else False) for each, as (train, val).

    Not foolproof; user can set flags manually or via callbacks. For further
    assurance, check `temp_history`, `val_temp_history`, and cache attributes
    (e.g. `_preds_cache`) which are cleared at end of :meth:`validate` by default;
    this method checks only flags: `_train_loop_done`, `train_postiter_processed`,
    `_val_loop_done`, `_val_postiter_processed`.
    """
    def _train_status():
        trained = True
        if not self._train_postiter_processed:
            trained = False
            if not self._train_loop_done:
                print("Incomplete or not called `_train_postiter_processing()` "
                      "within `train()`.")
            else:
                print("Improbable state; `_train_postiter_processed=True` is set "
                      "right before (and only before) `_train_loop_done=True`, "
                      "yet former is False.")
        return trained

    def _val_status():
        validated = True
        if self._val_loop_done:
            validated = False
            if not self._train_loop_done:
                print("NOTE: validating with `_train_loop_done==False`; OK if "
                      "`validate()` was called manually, else something went "
                      "wrong.")
            if self._val_postiter_processed:
                print("Incomplete or not called `_on_val_end()` within "
                      "`validate()`.")
            else:
                print("Improbable state; `_val_loop_done=True` is immediately "
                      "followed by `_val_postiter_processed=True`, yet latter is "
                      "False. Possibly inappropriate attribute setting by "
                      "callbacks.")
        elif not self._val_postiter_processed:
            validated = False
            print("Interrupted during validation loop within `validate()`; "
                  "incomplete or not called `_val_postiter_processing()`.")
        return validated

    trained = _train_status()
    validated = _val_status()

    if trained and validated:
        print("No interrupts detected.")
    elif trained:
        print("Interrupted: train[no], validation[yes].")
    elif validated:
        print("Interrupted: train[yes], validation[no].")
    else:
        print("Both train and validation were interrupted.")

    flags = ('_train_loop_done', '_train_postiter_processed',
             '_val_loop_done', '_val_postiter_processed')
    print(("\nFlags checked:" + "\n\t{} = {}" * len(flags)).format(
        *[x for f in flags for x in (f.ljust(max(map(len, flags))),
                                     getattr(self, f))]))
    return (not trained, not validated)


def info(self):
    """Prints various useful TrainGenerator & DataGenerator attributes,
    and interrupt status."""
    print("Epochs: %s/%s" % (self.epoch, self.epochs))
    train_total = len(self.datagen.set_nums_original)
    val_total   = len(self.val_datagen.set_nums_original)
    print("Train batches fit: %s/%s (in current epoch)\n"
          % (train_total - len(self.datagen.set_nums_to_process), train_total) +
          "Val   batches fit: %s/%s (in current validation)"
          % (val_total - len(self.val_datagen.set_nums_to_process), val_total)
          )
    print('-' * 80)

    print(("Best model directory: %s\n" % self.best_models_dir +
           "Checkpoint directory: %s\n" % self.logdir +
           "Load path: %s\n"            % self.loadpath +
           "Model full name: %s"        % self.model_name
           ))
    print('-' * 80)

    self.interrupt_status()
