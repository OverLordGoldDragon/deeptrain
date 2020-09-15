# -*- coding: utf-8 -*-
import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from types import LambdaType, MethodType

from .util._default_configs import _DEFAULT_TRAINGEN_CFG
from .util.configs  import _TRAINGEN_CFG
from .util._traingen_utils import TraingenUtils
from .util.logging  import _log_init_state
from .util.misc     import pass_on_error, capture_args
from .introspection import print_dead_weights, print_nan_weights
from .introspection import print_large_weights
from .callbacks     import TraingenCallback
from .util._backend import IMPORTS, Unbuffered, NOTE, WARN
from .backend import model_utils

sys.stdout = Unbuffered(sys.stdout)


class TrainGenerator(TraingenUtils):
    """The central DeepTrain class. Interfaces training, validation,
    checkpointing, data loading, and progress tracking.

    Arguments:
        model: models.Model / models.Sequential [keras / tf.keras]
            Compiled model to train.
        datagen: :class:`~deeptrain.data_generator.DataGenerator`
            Train data generator; fetches inputs and labels, handles
            preprocessing, shuffling, stateful formats, and informing
            TrainGenerator when a dataset is exhausted (epoch end).
        val_datagen: :class:`~deeptrain.data_generator.DataGenerator`
            Validation data generator.
        epochs: int
            Number of train epochs.
        logs_dir: str / None
            Path to directory where to generate log directories, that include
            TrainGenerator state, state report, model data, and others; see
            :meth:`~deeptrain.util.saving.checkpoint`. If `None`, will not
            checkpoint - but model saving still possible via `best_models_dir`.
        best_models_dir: str / None
            Path to directory where to save best model. "Best" means having new
            highest (`max_is_best==True`) or lowest (`max_is_best==False`)
            entry in `key_metric_history`.
            See :meth:`~deeptrain.util.saving._save_best_model`.
        loadpath: str / None
            Path to .h5 file containing TrainGenerator state to load (postfixed
            `'__state.h5'` by default). See :meth:`~deeptrain.util.saving.load`.
        callbacks: dict[str: function] /\
        :class:`~deeptrain.callbacks.TraingenCallback` / None
            Functions to apply at various stages, including training, validation,
            saving, loading, and `__init__`.
            See :class:`~deeptrain.callbacks.TraingenCallback`.
        fit_fn: str / `function(x, y, sample_weight)`
            Function, or name of model method to feed data to during training;
            if str, will define `fit_fn = getattr(model, 'fit')` (example).
            If function, its name (substring) must include `'fit'` or `'train'`
            (currently both function identically).
        eval_fn: str / `function(x, y, sample_weight)`
            Function, or name of model method to feed data to during validation;
            if str, will define `eval_fn = getattr(model, 'evaluate')` (example).
            If function, its name (substring) must include `'evaluate'` or
            `'predict'`:

                - `'evaluate'`: `eval_fn` uses data & labels to return metrics.
                - `'predict'`:  `eval_fn` uses data to return predictions, which
                  are used internally to compute metrics.

        key_metric: str
            Name of metric to track for saving best model; will store in
            `key_metric_history`.
            See :meth:`~deeptrain.util.saving._save_best_model`.
        key_metric_fn: function / None
            Custom function to compute key metric; overrides `key_metric` if
            not None.
        val_metrics: list[str] / None
            Names of metrics to track during validation.

                - If `'predict'` is not in `eval_fn.__name__`, is overridden by
                  model metrics (`model.compile(metrics=...)`)
                - If `'loss'` is not included, will prepend.
                - If `'*'` is included, will insert model metrics at its
                  position and pop `'*'`. Ex: `[*, 'f1_score']` ->
                  `['loss', 'accuracy', 'f1_score']`.

        custom_metrics: dict[str: function]
            Name-function pairs of custom functions to use for gathering metrics.
            Functions must obey `(y_true, y_pred)` input signature for first two
            arguments. They may additionally supply `sample_weight` and
            `pred_threshold`, which will be detected and used automatically.

                - Note: if using a custom metric in `model.compile(loss=tf_fn)`,
                  name in `custom_metrics` must be function's code name, i.e.
                  `{tf_fn.__name__: fn}` (where `fn` is e.g. numpy version).

        input_as_labels: bool
            Feed model input also to its output. Ex: autoencoders.
        max_is_best: bool
            Whether to consider greater `key_metric` as better in saving best
            model. See :meth:`~deeptrain.util.saving._save_best_model`.
            If None, defaults to False if `key_metric=='loss'`, else True.
        val_freq: None / dict[str: int], str in {'iter', 'batch', 'epoch', 'val'}
            How frequently to validate. `{'epoch': 1}` -> every epoch;
            `{'iter': 24}` -> every 24 train iterations. Only one key-value
            pair supported. If None, won't validate.
        plot_history_freq: None / dict[str: int], str in {'iter', 'batch', \
        'epoch', 'val'}
            How frequently to plot train & validation history. Only one key-value
            pair supported. If None, won't plot history.
        unique_checkpoint_freq: None / dict[str: int], str in {'iter', \
        'batch', 'epoch', 'val'}
            How frequently to make checkpoints with unique savefile names, as
            opposed to temporary ones which are overwritten each time. Only one
            key-value pair supported. If None, won't make unique checkpoints.
        temp_checkpoint_freq: None / dict[str: int], str in {'iter', 'batch', \
        'epoch', 'val'}
            How frequently to make checkpoints with the same predefined name,
            to be overwritten ("temporary"); serves as an intermediate
            checkpoint to unique ones, if needed. Only one key-value pair
            supported. If None, won't make temporary checkpoints.
        class_weights: dict[int: int] / None
            Integer-mapping of class labels to their "weights"; if not None,
            will feed `sample_weight` mediated by the weights to train function
            (`fit_fn`).

            >>> class_weights = {0: 4, 1: 1}
            >>> labels        == [1, 1, 0, 1]  # if
            >>> sample_weight == [4, 4, 1, 4]  # then

        val_class_weights: dict[int: int] / None
            `class_weights` for validation function (`eval_fn`).
        reset_statefuls: bool
            Whether to call `model.reset_states()` at the end of every batch
            (train and val).
        iter_verbosity: int in {0, 1, 2}
            - 0: print no iteration info
            - 1: print name of set being fit / validated, metric names and values,
                 and `model.reset_states()` being called
            - 2: print a `'.'` at every iteration (useful if having multiple
              iterations per batch)
        logdir: str / None
            Directory where to write logs to (see `logs_dir`). Use to specify
            an existing directory (to, for example, resume training and logging
            in original folder). Overrides `logs_dir`.
        optimizer_save_configs: dict / None
            Dict specifying which optimizer attributes to include or exclude
            when saving. See :meth:`~deeptrain.util.saving.save`.
        optimizer_load_configs: dict / None
            Dict specifying which optimizer attributes to include or exclude
            when loading. See :meth:`~deeptrain.util.saving.load`.
        plot_configs: dict / None
            Dict specifying :meth:`~deeptrain.visuals.get_history_fig` behavior.
            See :data:`~deeptrain.util._default_configs._DEFAULT_PLOT_CFG`, and
            :meth:`~deeptrain.util.misc._make_plot_configs_from_metrics`.
        model_configs: dict / None
            Dict specifying model information. Intended usage is: create `model`
            according to the dict, specifying hyperparameters, loss function, etc.

            >>> def make_model(batch_shape, units, optimizer, loss):
            ...     ipt = Input(batch_shape=batch_shape)
            ...     out = Dense(units)(ipt)
            ...     model = Model(ipt, out)
            ...     model.compile(optimizer, loss)
            ...     return model
            ...
            >>> model_configs = {'batch_shape': (32, 16), 'units': 8,
            ...                  'optimizer': 'adam', 'loss': 'mse'}
            >>> model = make_model(**model_configs)

            Checkpoints will include an image report with the entire dict;
            the larger the portion of the model that's created according to
            `model_configs`, the more will be documented for easy reference.
        kwargs: keyword arguments.
            See :data:`~deeptrain.util._default_configs._DEFAULT_TRAINGEN_CFG`.
            `kwargs` and all other arguments are subject to validation and
            correction by
            :meth:`~deeptrain.util.misc._validate_traingen_configs`.

    `__init__`:

    Instantiation. ("+" == if certain conditions are met)

        - Pulls methods from
          :class:`~deeptrain.util._traingen_utils.TraingenUtils`
        - Validates args & kwargs, and tries to correct, printing a"NOTE" or
          "WARNING" message where appropriate
        - +Instantiates logging directory
        - +Loads `TrainGenerator`, `datagen`, and `val_datagen` states
        - +Loads model and optimizer weights (but not model architecture)
        - +Preloads train & validation data (before a call to :meth:`train`
          is made).
        - +Applies initial callbacks
        - +Logs initial state (:meth:`~deeptrain.util.logging._log_init_state`)
        - Captures and saves all arguments passed to `__init__`
        - Instantiates misc internal parameters to predefiend values (may be
          overridden by loading).
    """
    @capture_args
    def __init__(self, model, datagen, val_datagen,
                 epochs=1,
                 logs_dir=None,
                 best_models_dir=None,
                 loadpath=None,
                 callbacks=None,

                 fit_fn='train_on_batch',
                 eval_fn='evaluate',
                 key_metric='loss',
                 key_metric_fn=None,
                 val_metrics=None,
                 custom_metrics=None,
                 input_as_labels=False,
                 max_is_best=None,

                 val_freq={'epoch': 1},
                 plot_history_freq={'epoch': 1},
                 unique_checkpoint_freq={'epoch': 1},
                 temp_checkpoint_freq=None,

                 class_weights=None,
                 val_class_weights=None,

                 reset_statefuls=False,
                 iter_verbosity=1,
                 logdir=None,
                 optimizer_save_configs=None,
                 optimizer_load_configs=None,
                 plot_configs=None,
                 model_configs=None,
                 **kwargs):
        super().__init__()

        self.model=model
        self.datagen=datagen
        self.val_datagen=val_datagen
        self.epochs=epochs
        self.logs_dir=logs_dir
        self.best_models_dir=best_models_dir
        self.loadpath=loadpath
        self.callbacks=callbacks or []

        #### metrics ##########################################################
        self.key_metric=key_metric
        self.key_metric_fn=key_metric_fn
        self.train_metrics = model_utils.get_model_metrics(model)
        self.val_metrics=val_metrics
        self.custom_metrics=custom_metrics or {}
        self.input_as_labels=input_as_labels
        if max_is_best is None:
            self.max_is_best = False if self.key_metric == 'loss' else True
        else:
            self.max_is_best = max_is_best

        #### internal callback frequencies ####################################
        self.val_freq=val_freq
        self.plot_history_freq=plot_history_freq
        self.unique_checkpoint_freq=unique_checkpoint_freq
        self.temp_checkpoint_freq=temp_checkpoint_freq

        #### misc #############################################################
        self.class_weights=class_weights
        self.val_class_weights=val_class_weights

        self.reset_statefuls=reset_statefuls
        self.iter_verbosity=iter_verbosity
        self.logdir=logdir
        self.optimizer_save_configs=optimizer_save_configs
        self.optimizer_load_configs=optimizer_load_configs
        self.plot_configs=plot_configs
        self.model_configs = model_configs
        self.batch_size=kwargs.pop('batch_size', None) or model.output_shape[0]

        self.fit_fn=fit_fn    # uses @property.setter
        self.eval_fn=eval_fn  # uses @property.setter

        #### loading, logging, callbacks, kwargs init #########################
        self._passed_args = kwargs.pop('_passed_args', None)
        self._init_and_validate_kwargs(kwargs)
        self._init_class_vars()

        self._init_callbacks_called = False
        if self.loadpath:
            self.load(passed_args=self._passed_args)
        else:
            self._prepare_initial_data()
            self._init_callbacks()  # called in `load`

        if self.logdir or self.logs_dir:
            self._init_logger()
        else:
            self.logdir = None
            print(NOTE, "logging OFF")

        if self.logdir:
            savedir = os.path.join(self.logdir, "misc")
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
            pass_on_error(_log_init_state, self, kwargs, savedir=savedir,
                          errmsg=WARN + " could not log init state - skipping")

    ###### MAIN METHODS #######################################################
    def train(self):
        """The train loop.

            - Fetches data from `get_data`
            - Fits data via `fin_fn`
            - Processes fit metrics in `_train_postiter_processing`
            - Stores metrics in `history`
            - Applies `'train:iter'`, `'train:batch'`, and `'train:epoch'`
              callbacks
            - Calls `validate` when appropriate

        **Interruption**:

            - *Safe*: during `get_data`, which can be called indefinitely
              without changing any attributes.
            - *Avoid*: during `_train_postiter_processing`, where `fit_fn` is
              applied and weights are updated - but metrics aren't stored, and
              `_train_postiter_processed=False`, restarting the loop without
              recording progress.
            - Best bet is during :meth:`validate`, as `get_data` may be too brief.
        """
        while self.epoch < self.epochs:
            if not self._train_loop_done:
                if self._train_postiter_processed:
                    x, y, sample_weight = self.get_data(val=False)
                    sw = sample_weight if self.class_weights else None
                    if self.iter_verbosity:
                        self._print_iter_progress()

                    self._metrics_cached = self.fit_fn(x, y, sw)
                    # `_metrics_cached` in case of interrupt

                self._train_postiter_processed = False
                self._train_postiter_processing(self._metrics_cached)
                self._train_postiter_processed = True
            else:
                self.validate()
                self.train()

        print("Training has concluded.")

    def validate(self, record_progress=True, clear_cache=True, restart=False,
                 use_callbacks=True):
        """Validation loop.

            - Fetches data from `get_data`
            - Applies function based on `_eval_fn_name`
            - Processes and caches metrics/predictions in
              `_val_postiter_processing`
            - Applies `'val:iter'`, `'val:batch'`, and `'val:epoch'` callbacks
            - Calls `_on_val_end` at end of validation to compute metrics
              and store them in `val_history`
            - Applies `'val_end'` and maybe `('val_end': 'train:epoch')` callbacks
            - If `restart`, calls :meth:`reset_validation`.

        **Arguments**:
            record_progress: bool
                If False, won't update `val_history`, `_val_iters`,
                `_batches_validated`.
            clear_cache: bool
                If False, won't call :meth:`clear_cache`; useful for keeping
                preds & labels acquired during validation.
            restart: bool
                If True, will call :meth:`reset_valiation` before validation loop
                to reset validation attributes; useful for starting afresh (e.g.
                if interrupted).
            use_callbacks: bool
                If False, won't call :meth:`apply_callbacks`
                or :meth:`plot_history`.

        **Interruption:**

            - *Safe*: during `get_data`, which can be called indefinitely
              without changing any attributes.
            - *Avoid*: during `_val_postiter_processing`. Model remains
              unaffected*, but caches are updated; a restart may yield duplicate
              appending, which will error or yield inaccuracies.
              (* forward pass may consume random seed if random ops are used)
            - *In practice*: prefer interrupting immediately after
              `_print_iter_progress` executes.
        """
        if restart:
            self.reset_validation()
        print("\n\nValidating..." if not self._val_loop_done else
              "\n\nFinishing post-val processing...")

        while not self._val_loop_done:
            data = {}
            if self._val_postiter_processed:
                x, self._y_true, self._val_sw = self.get_data(val=True)
                sw = self._val_sw if self.val_class_weights else None
                if self.iter_verbosity:
                    self._print_iter_progress(val=True)

                data['metrics'] = self.eval_fn(x, self._y_true, sw)
                data['batch_size'] = len(x)

            self._val_postiter_processed = False
            self._val_postiter_processing(record_progress, use_callbacks, **data)
            self._val_postiter_processed = True

        if self._val_loop_done:
            self._on_val_end(record_progress, use_callbacks, clear_cache)

    ###### MAIN METHOD HELPERS ################################################
    def _train_postiter_processing(self, metrics):
        """Procedures done after every train iteration. Similar to
        :meth:`_val_postiter_processing`, except operating on train rather than
        val variables, and calling :meth:`validate` when appropriate.
        """
        def _on_iter_end(metrics):
            self._update_temp_history(metrics)
            self._fit_iters += 1
            self.datagen.update_state()
            self._apply_callbacks(stage='train:iter')

        def _on_batch_end():
            self._train_new_batch_notified = False

            self._batches_fit += 1
            self._train_x_ticks.append(self._batches_fit)
            self._train_val_x_ticks.append(self._times_validated)
            self._set_name_cache.append(self._set_name)
            try:
                self._update_train_history()
                if self.iter_verbosity >= 1:
                    self._print_train_progress()
            except Exception as e:
                # might happen due to incomplete loaded data for updating history
                print(WARN, "could not update and print progress"
                      "- OK if right after load; skipping...\nErrmsg: %s" % e)

            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset ', end='')
            self._apply_callbacks(stage='train:batch')

        def _on_epoch_end():
            self.temp_history = deepcopy(self._temp_history_empty)
            self.epoch = self.datagen.on_epoch_end()

            overline = "_" * len(f" EPOCH {self.epoch} -- COMPLETE ")
            decor = "\n{}\n\033[4m {}{}{} \033[0m\n"
            print(decor.format(overline, "EPOCH ", self.epoch, " -- COMPLETE"))

            self._hist_vlines     += [self._batches_fit]
            self._val_hist_vlines += [self._times_validated]
            self._apply_callbacks(stage='train:epoch')

        def _should_validate():
            return self._should_do(self.val_freq)

        _on_iter_end(metrics)
        if self.datagen.batch_exhausted:
            _on_batch_end()
        if self.datagen.all_data_exhausted:
            _on_epoch_end()

        if _should_validate():
            self._train_postiter_processed = True  # in case val is interrupted
            self._train_loop_done = True
            self._val_loop_done = False
            self.validate()

    def _val_postiter_processing(self, record_progress=True, use_callbacks=True,
                                 metrics=None, batch_size=None):
        """Procedures done after every validation iteration. Unless marked
        "always", are conditional and may skip.

            - Update temp val history (always)
            - Update `val_datagen` state (always)
            - Update val cache (preds, labels, etc)
            - Update val history
            - Reset statefuls
            - Print progress
            - Apply callbacks

        Executes internal "callbacks" when appropriate: `_on_iter_end`,
        `_on_batch_end`, `_on_epoch_end`. List not exhaustive.
        """
        def _on_iter_end(metrics, batch_size):
            if 'predict' in self._eval_fn_name:
                self._y_preds = metrics
            elif 'evaluate' in self._eval_fn_name:
                self._update_temp_history(metrics, val=True)

            if record_progress:
                self._val_iters += 1

            if self.batch_size is None:
                if self._inferred_batch_size is None:
                    self._inferred_batch_size = batch_size
                elif self._inferred_batch_size != batch_size:
                    self._inferred_batch_size = 'varies'

            if 'predict' in self._eval_fn_name:
                self._update_val_iter_cache()
            self.val_datagen.update_state()

            if use_callbacks:
                self._apply_callbacks(stage='val:iter')

        def _on_batch_end():
            if record_progress:
                self._batches_validated += 1
            self._val_set_name_cache.append(self._val_set_name)

            if self.iter_verbosity >= 1:
                self._print_val_progress()
            self._val_new_batch_notified = False

            if self.reset_statefuls:
                self.model.reset_states()
                if self.iter_verbosity >= 1:
                    print('RNNs reset', end=' ')
            if use_callbacks:
                self._apply_callbacks(stage='val:batch')

        def _on_epoch_end():
            if record_progress:
                self._update_val_history()
            self.val_epoch = self.val_datagen.on_epoch_end()
            if use_callbacks:
                self._apply_callbacks(stage='val:epoch')
            self._val_loop_done = True

        _on_iter_end(metrics, batch_size)
        if self.val_datagen.batch_exhausted:
            _on_batch_end()
        if self.val_datagen.all_data_exhausted:
            _on_epoch_end()


    def _on_val_end(self, record_progress, use_callbacks, clear_cache):
        """Procedures done after :meth:`~validate`. Unless marked "always", are
        conditional and may skip. List not exhaustive.

            - Update train/val history
            - Clear cache
            - Plot history
            - Checkpoint
            - Apply callbacks
            - Check model health
            - Validate `batch_size` (always)
            - Reset validation flags: `_inferred_batch_size`, `_val_loop_done`,
              `_train_loop_done` (always)
        """
        def _record_progress():
            self._times_validated += 1
            self.val_epoch = self.val_datagen.on_epoch_end()
            self._val_x_ticks += [self._times_validated]
            self._val_train_x_ticks += [self._batches_fit]

            if self.max_is_best:
                new_best = (self.key_metric_history[-1] > self.best_key_metric)
            else:
                new_best = (self.key_metric_history[-1] < self.best_key_metric)

            if new_best and self.best_models_dir is not None:
                self._save_from_on_val_end = True  # flag used in `save()`
                self._save_best_model(del_previous_best=self.max_one_best_save)

            if self.logdir:
                do_temp = self._should_do(self.temp_checkpoint_freq)
                do_unique = self._should_do(self.unique_checkpoint_freq)
                if do_temp or do_unique:
                    self._save_from_on_val_end = True  # flag used in `save()`
                    self.checkpoint()

        def _print_best_subset():
            best_nums = ", ".join(map(str, self.best_subset_nums))
            best_size = self.best_subset_size
            print("Best {}-subset: {}".format(best_size, best_nums))

        def _validate_batch_size():
            batch_size = self.batch_size or self._inferred_batch_size
            if (not isinstance(batch_size, int) and
                'predict' in self._eval_fn_name):
                raise ValueError(
                    "to use `'predict' in _eval_fn_name`, either (1) `batch_size`"
                    " must be defined, or (2) data fed in `validation()` "
                    "must have same len() / .shape[0] across iterations.")

        _validate_batch_size()
        if self.best_subset_size:
            _print_best_subset()
        if record_progress:
            _record_progress()

        if self._should_do(self.plot_history_freq) and use_callbacks:
            pass_on_error(self.plot_history, update_fig=record_progress,
                          errmsg=(WARN + " model history could not be "
                                  "plotted; skipping..."))

        if use_callbacks:
            if self.datagen.all_data_exhausted:
                self._apply_callbacks(stage=('val_end', 'train:epoch'))
            else:
                self._apply_callbacks(stage='val_end')

        if clear_cache:
            self.clear_cache()
        if self.check_model_health:
            self.check_health()

        self._inferred_batch_size = None  # reset
        self._val_loop_done = False
        self._train_loop_done = False

    def get_data(self, val=False):
        """Get train (`val=False`) or validation (`val=True`) data from
        `datagen` or `val_datagen`, respectively.
        See :class:`~deeptrain.data_generator.DataGenerator`.

        `DataGenerator.get()` returns `x, labels`; if `input_as_data == True`,
        sets `y = x` - else, `y = labels`. Either way, sets
        `class_labels = labels`. Generates `sample_weight` from `class_labels`.
        """
        def _standardize_shape(labels):
            class_labels = labels
            # broadcast `labels` to same rank as `model.output_shape`
            while len(class_labels.shape) < len(self.model.output_shape):
                class_labels = np.expand_dims(class_labels, -1)
            return class_labels

        datagen = self.val_datagen if val else self.datagen
        if datagen.batch_exhausted:
            datagen.advance_batch()
            setattr(self, '_val_set_name' if val else '_set_name',
                    datagen.set_name)

        x, labels = datagen.get()
        y = labels if not self.input_as_labels else x

        if isinstance(labels, (np.ndarray, list)) and len(labels) > 0:
            class_labels = _standardize_shape(labels)
            slice_idx = getattr(datagen, 'slice_idx', None)
            sample_weight = self.get_sample_weight(class_labels, val, slice_idx)
        else:
            sample_weight = np.ones(len(x))

        return x, y, sample_weight

    def clear_cache(self, reset_val_flags=False):  # to `validate` from scratch
        """Call to reset cache attributes accumulated during validation; useful
        for "restarting" validation (before calling :meth:`validate`).

        Attributes set to `[]`: `{'_preds_cache', '_labels_cache', '_sw_cache',
        '_class_labels_cach', '_set_name_cache', '_val_set_name_cach',
        '_y_true', '_val_sw'}`.
        """
        attrs_to_clear = ('_preds_cache', '_labels_cache', '_sw_cache',
                          '_class_labels_cache',
                          '_set_name_cache', '_val_set_name_cache',
                          '_y_true', '_val_sw')
        [setattr(self, attr, []) for attr in attrs_to_clear]
        self.val_temp_history = deepcopy(self._val_temp_history_empty)

        if reset_val_flags:
            self._inferred_batch_size = None
            self._val_loop_done = False
            self._train_loop_done = False
            self._val_postiter_processed = True

    def reset_validation(self):
        """Used to restart validation (e.g. in case interrupted); calls
        :meth:`clear_cache` and :meth:`DataGenerator.reset_state`
        (and, if `reset_statefuls`, `model.reset_states()`).

        Does not reset validation counters (e.g. `_val_iters`).
        """
        self.clear_cache(reset_val_flags=True)
        self.val_datagen.reset_state(shuffle=False)
        if self.reset_statefuls:
            self.model.reset_states()

    def _should_do(self, freq_config, forced=False):
        """Checks whether a counter meets a frequency as specified in
        `val_freq`, `plot_history_freq`, `unique_checkpoint_freq`,
        `temp_checkpoint_freq`.

        "Counter" is one of `_fit_iters`, `_batches_fit`, `epoch`, and
        `_times_validated`. Ex: with `unique_checkpoint_freq = {'batch': 5}`,
        :meth:`checkpoint` will make a unique checkpoint on every 5th batch
        fitted during :meth:`.train`.
        """
        if forced:
            return True
        if freq_config is None:
            return False
        freq_mode, freq_value = list(freq_config.items())[0]

        if freq_mode == 'iter':
            return self._fit_iters % freq_value == 0
        elif freq_mode == 'batch':
            batch_done = self.datagen.batch_exhausted
            return (self._batches_fit % freq_value == 0) and batch_done
        elif freq_mode == 'epoch':
            epoch_done = self.datagen.all_data_exhausted
            return (self.epoch % freq_value == 0) and epoch_done
        elif freq_mode == 'val':
            return (self._times_validated % freq_value == 0)

    ###### LOG METHODS ########################################################
    def _update_val_iter_cache(self):
        """Called by `_on_iter_end` within :meth:`_val_postiter_processing`;
        updates validation cache variables (`_labels_cache`, `_preds_cache`,
        `_class_labels_cache`, `_sw_cache`).

        If `val_datagen` has a non-None `slice_idx`, will preserve batch-slice
        structure:

        >>> [[y00, y01, y02], [y10, y11, y12]]    # 2 batches, 3 slices/batch
        >>> [[y00, y01], [y10, y11], [y20, y21]]  # 3 batches, 2 slices/batch
        """
        def _standardize_shapes(*data):
            # ensure shapes are in format expected by methods in util.training
            ls = []
            for x in data:
                if isinstance(x, np.ndarray):  # could be empty list
                    while len(x.shape) < len(self.model.output_shape):
                        x = np.expand_dims(x, -1)
                ls.append(x)
            return ls

        y, class_labels, sample_weight = _standardize_shapes(
            self._y_true, self.val_datagen.labels, self._val_sw)
        if (class_labels is None or
            (isinstance(class_labels, list) and len(class_labels) == 0)):
            # training.py expects batch_size-sized arrays for `class_labels`
            class_labels = np.zeros(len(y))

        if getattr(self.val_datagen, 'slice_idx', None) is None:
            self._sw_cache.append(sample_weight)
            self._preds_cache.append(self._y_preds)
            self._labels_cache.append(y)
            self._class_labels_cache.append(class_labels)
            return

        if getattr(self.val_datagen, 'slice_idx', None) == 0:
            # if using sliced batches, append a container to accumulate
            # data for the multiple iterations (slices) per batch
            self._labels_cache.append([])
            self._class_labels_cache.append([])
            self._sw_cache.append([])
            if 'predict' in self._eval_fn_name:
                self._preds_cache.append([])

        self._sw_cache[-1].append(sample_weight)
        self._labels_cache[-1].append(y)
        self._class_labels_cache[-1].append(class_labels)
        if 'predict' in self._eval_fn_name:
            self._preds_cache[-1].append(self._y_preds)

    def _update_val_history(self):
        for name, metric in self._get_val_history().items():
            self.val_history[name].append(metric)
        self.key_metric_history.append(self.val_history[self.key_metric][-1])

    def _get_train_history(self):
        return {metric:np.mean(values) for metric, values
                in self.temp_history.items()}

    def _update_train_history(self):
        for metric, value in self._get_train_history().items():
            self.history[metric] += [value]

    def _print_train_progress(self):
        """Called within :meth:`_train_postiter_processing`, by `on_batch_end()`.
        """
        train_metrics = self._get_train_history()
        for name in self.metric_printskip_configs.get('train', []):
            train_metrics.pop(name, None)
        self._print_progress(train_metrics, endchar='')

    def _print_val_progress(self):
        """Called within :meth:`_val_postiter_processing`, by `on_batch_end()`.
        """
        val_metrics = self._get_val_history(for_current_iter=True)
        for name in self.metric_printskip_configs.get('val', []):
            val_metrics.pop(name, None)
        self._print_progress(val_metrics)

    def _print_progress(self, metrics, endchar='\n'):
        """Called by :meth:`_print_train_progress` and
        :meth:`_print_val_progress`."""
        names  = [self._metric_name_to_alias(name) for name in metrics]
        values = [v for v in metrics.values()]
        assert len(names) == len(values)

        names_joined  = ', '.join(names)
        values_joined = ', '.join([('%.6f' % v) for v in values])
        if len(names) != 1:
            names_joined  = '(%s)' % names_joined
            values_joined = '(%s)' % values_joined
        print(" {} = {} ".format(names_joined, values_joined), end=endchar)

    def _print_iter_progress(self, val=False):
        """Called within :meth:`train` and :meth:`validate`."""
        if val:
            if not self._val_new_batch_notified:
                pad = self._val_max_set_name_chars + 3
                padded_num_txt = (self._val_set_name + "...").ljust(pad)
                print(end="Validating set %s" % padded_num_txt)
                self._val_new_batch_notified = True
        else:
            if not self._train_new_batch_notified:
                pad = self._max_set_name_chars + 3
                padded_num_txt = (self._set_name + "...").ljust(pad)
                print(end="\nFitting set %s" % padded_num_txt)
                self._train_new_batch_notified = True
        if self.iter_verbosity >= 2:
            print(end='.')

    ###### VISUAL/CALC METHODS ################################################
    def plot_history(self, update_fig=True, w=1, h=1):
        """Plots train & validation history (from `history` and `val_history`).

        - `update_fig=True` -> store latest fig in `_history_fig`.
        - `w` & `h` scale the width & height, respectively, of the figure.
        - Plots configured by `plot_configs`.
        """
        def _show_closed_fig(fig):
            _fig = plt.figure()
            manager = _fig.canvas.manager
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
            plt.show()

        fig = self.get_history_fig(self.plot_configs, w, h)
        if update_fig:
            self._history_fig = fig
        _show_closed_fig(fig)

    ###### CALLBACK METHODS ###################################################
    def _apply_callbacks(self, stage):
        """Callbacks. See examples/callbacks

        Two approaches:
            1. Class-based: inherit deeptrain.callbacks.TraingenCallback,
               define stage-based methods, e.g. on_train_epoch_end. Methods
               also take `stage` argument for further control, e.g. to only
               call `on_train_epoch_end` when
               `stage == ('val_end', 'train:epoch')`.
            2. Function-based: make a dict of stage-function call pairs, e.g.:

               >>> {'train:epoch': (fn1, fn2),
               ... 'val_batch': fn3,
               ... ('val_end': 'train:epoch'): fn4}

               Callback will execute if a key is `in` the `stage` passed to
               `_apply_callbacks`; e.g. `(fn1, fn2)` will execute on
               `stage==('val_end', 'train:epoch')`, with key `'train:epoch'`,
               but `fn4` won't execute, on `stage=='train:epoch'`.
        """
        def _matched(cb_stage, stage):
            if isinstance(cb_stage, tuple):
                if isinstance(stage, tuple):
                    return cb_stage == stage
                return all(cs == stage for cs in cb_stage)
            return cb_stage in stage

        if not hasattr(self, '_cb_alias'):
            self._cb_alias = {'train:iter':  'on_train_iter_end',
                              'train:batch': 'on_train_batch_end',
                              'train:epoch': 'on_train_epoch_end',
                              'val:iter':    'on_val_iter_end',
                              'val:batch':   'on_val_batch_end',
                              'val:epoch':   'on_val_epoch_end',
                              'val_end':     'on_val_end',
                              'save':        'on_save',
                              'load':        'on_load'}

        for cb in self.callbacks:
            if isinstance(cb, TraingenCallback):
                _stage = stage if not isinstance(stage, tuple) else stage[0]
                fn = getattr(cb, self._cb_alias[_stage])
                try:
                    fn(stage)
                except NotImplementedError:
                    pass
            elif isinstance(cb, dict):
                for cb_stage in cb:
                    if _matched(cb_stage, stage):
                        for fn in cb[cb_stage]:
                            fn(self)
            else:
                raise TypeError("unsupported callback type: %s" % type(cb)
                                + "; must be either dict, or subclass "
                                "TraingenCallback.")

    def _init_callbacks(self):
        """Instantiates callback objects (must subclass
        :class:`~callbacks.TraingenCallback`), passing in `TrainGenerator`
        instance as first (and only) argument. Enables custom callbacks
        utilizing `TrainGenerator` attributes and methods.
        """
        for cb in self.callbacks:
            if isinstance(cb, TraingenCallback):
                try:
                    cb.init_with_traingen(self)
                except NotImplementedError:
                    pass
        self._init_callbacks_called = True

    ###### MISC METHODS #######################################################
    def check_health(self, dead_threshold=1e-7, dead_notify_above_frac=1e-3,
                     large_threshold=3, large_notify_above_frac=1e-3,
                     notify_detected_only=True):
        """Check whether any layer weights have 'zeros' or NaN weights;
        very fast / inexpensive.

        Arguments:
            dead_threshold: float
                Count values below this as zeros.
            dead_notify_above_frac: float
                If fraction of values exceeds this, print it and
                the weight's name.
            notify_detected_only: bool
                True  -> print only if dead/NaN found
                False -> print a 'not found' message
        """
        print_dead_weights(self.model, dead_threshold, dead_notify_above_frac,
                           notify_detected_only)
        print_nan_weights(self.model, notify_detected_only)
        print_large_weights(self.model, large_threshold, large_notify_above_frac,
                            notify_detected_only)

    def _alias_to_metric_name(self, alias):
        return self.alias_to_metric.get(alias.lower(), alias)

    def _metric_name_to_alias(self, metric_name):
        return self.metric_to_alias.get(metric_name.lower(), metric_name)

    def destroy(self, confirm=False, verbose=1):
        """Class 'destructor'. Sets own, datagen's, and val_datagen's attributes
        to `[]` (which can free memory of arrays), then deletes them. Also deletes
        'model' attribute, but this has no effect on memory allocation until
        it's dereferenced globally and the TensorFlow/Keras graph is cleared
        (best bet is to restart the Python kernel).
        """
        def _destroy():
            for obj in (self.datagen, self.val_datagen, self):
                attrs = list(vars(obj))
                for attr in attrs:
                    setattr(obj, attr, [])
                    delattr(obj, attr)
                del obj
            gc.collect()
            if verbose:
                print(">>>TrainGenerator DESTROYED")

        if confirm:
            _destroy()
            return
        response = input("!! WARNING !!\nYou are about to destroy TrainGenerator"
                         "; this will wipe all its own and DataGenerator's "
                         "(train & val) shallow-referenced data, and delete "
                         "respective attributes. `model` will be dereferenced, "
                         "but not destroyed. Proceed? [y/n]")
        if response == 'y':
            _destroy()

    ###### PROPERTIES #########################################################
    @property
    def fit_fn(self):
        return self._fit_fn

    @fit_fn.setter
    def fit_fn(self, fn):
        """Ensures fn name and args are changed when fn itself is set."""
        self._attr_fn_setter(fn, 'fit_fn',
                             supported_fn_names=('fit', 'train'))

    @property
    def eval_fn(self):
        return self._eval_fn

    @eval_fn.setter
    def eval_fn(self, fn):
        """Ensures fn name and args are changed when fn itself is set."""
        self._attr_fn_setter(fn, 'eval_fn',
                             supported_fn_names=('evaluate', 'predict'))

    def _attr_fn_setter(self, fn, attr_name, supported_fn_names):
        def _make_tf_keras_fn(model, name):
            _fn = getattr(model, name)
            if name == 'train_on_batch':
                def fn(x, y, sw):
                    return _fn(x, y, sample_weight=sw)
            elif name == 'predict_on_batch':
                def fn(x, y, sw):
                    return _fn(x)
            elif name == 'predict':
                def fn(x, y, sw):
                    return _fn(x, batch_size=len(x))
            elif name in {'evaluate', 'fit'}:
                def fn(x, y, sw):
                    return _fn(x, y, sample_weight=sw, batch_size=len(x),
                               verbose=0)
            return fn

        if not isinstance(fn, (LambdaType, MethodType, str)):
            raise TypeError(f"'{attr_name}' must be a function, method, or name "
                            f"(str) of to fetch from `model` (got: {fn})")

        if isinstance(fn, str):
            name = fn
            fn = _make_tf_keras_fn(self.model, name)
        else:
            name = getattr(fn, '__qualname__', '') or fn.__name__
            name = name.split('.')[-1]  # drop packages / modules / classes
        if not any(s in name for s in supported_fn_names):
            raise ValueError(f"set `{attr_name}` with unsupported name; must "
                             "contain one of: " + ", ".join(supported_fn_names))

        setattr(self, f'_{attr_name}', fn)
        setattr(self, f'_{attr_name}_name', name)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        """Ensures `epoch` is also set in corresponding `DataGenerator`."""
        self._epoch = value
        self.datagen.epoch = value

    @property
    def val_epoch(self):
        return self._val_epoch

    @val_epoch.setter
    def val_epoch(self, value):
        """Ensures `epoch` is also set in corresponding `DataGenerator`."""
        self._val_epoch = value
        self.val_datagen.epoch = value

    ###### INIT METHODS #######################################################
    def _prepare_initial_data(self, from_load=False):
        """Preloads first batch for training and validation, and superbatch
        if available."""
        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            if dg.superbatch_set_nums or dg.superbatch_path:
                dg.preload_superbatch()
            if from_load:
                dg.batch_loaded = False  # load() might've set to True
                if dg.labels_path:       # load() might've changed `labels_path`
                    dg.preload_all_labels()
            dg.advance_batch()

            pf = '_val' if 'val' in dg_name else ''  # attr prefix
            setattr(self, pf + '_set_name', dg.set_name)
            setattr(self, pf + '_set_num', dg.set_num)

            print("%s initial data prepared" % ("Train" if not pf else "Val"))

    def _init_logger(self):
        """Instantiate log directory for checkpointing. If `logdir` was
        provided at `__init__`, will use it - else, will make a directory
        and assign its absolute path to `logdir`.
        """
        if not self.logdir:
            self.model_name = self.get_unique_model_name()
            self.model_num = int(self.model_name.split('__')[0].replace('M', ''))
            self.logdir = os.path.join(self.logs_dir, self.model_name)
            os.mkdir(self.logdir)
            print("Logging ON; directory (new):", self.logdir)
        else:
            print("Logging ON; directory (existing):", self.logdir)

    def _init_and_validate_kwargs(self, kwargs):
        """Sets and validates `**kwargs`, raising exception if kwargs result
        in an invalid configuration, or correcting them (and possibly notifying)
        when possible. Also catches unused arguments."""
        def _validate_kwarg_names(kwargs):
            for kw in kwargs:
                if kw not in _DEFAULT_TRAINGEN_CFG:
                    raise ValueError("unknown kwarg: '{}'".format(kw))

        def _set_kwargs(kwargs):
            class_kwargs = deepcopy(_TRAINGEN_CFG)
            class_kwargs.update(kwargs)

            for attribute in class_kwargs:
                setattr(self, attribute, class_kwargs[attribute])

        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)
        self._validate_traingen_configs()

    def _init_class_vars(self):
        """Instantiates various internal attributes. Most of these are saved
        and loaded by default."""
        #### init misc attributes ############################################
        self.best_key_metric=0 if self.max_is_best else 999
        self.epoch=0
        self.val_epoch=0
        self._set_name=None
        self._val_set_name=None
        self.model_name=self.get_unique_model_name()  # model_num set internally
        self._imports = IMPORTS.copy()

        self._history_fig=None
        self._times_validated=0
        self._batches_fit=0
        self._batches_validated=0
        self._fit_iters=0
        self._val_iters=0
        self._train_loop_done=False
        self._val_loop_done=False
        self._train_postiter_processed=True
        self._val_postiter_processed=True
        self._train_new_batch_notified=False
        self._val_new_batch_notified=False
        self._inferred_batch_size=None
        self._save_from_on_val_end=False

        as_empty_list = [
            'key_metric_history', 'best_subset_nums', '_labels',
            '_preds_cache', '_labels_cache', '_sw_cache',
            '_class_labels_cache',
            '_set_name_cache', '_val_set_name_cache',
            '_hist_vlines', '_val_hist_vlines',
            '_train_x_ticks', '_train_val_x_ticks',
            '_val_x_ticks', '_val_train_x_ticks',
            ]
        [setattr(self, name, []) for name in as_empty_list]

        #### init histories ###################################################
        self.history          = {name: [] for name in self.train_metrics}
        self.temp_history     = {name: [] for name in self.train_metrics}
        self.val_history      = {name: [] for name in self.val_metrics}
        self.val_temp_history = {name: [] for name in self.val_metrics}
        self._temp_history_empty     = deepcopy(self.temp_history)
        self._val_temp_history_empty = deepcopy(self.val_temp_history)

