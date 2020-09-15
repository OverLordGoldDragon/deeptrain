# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import deeptrain.metrics

from types import LambdaType
from functools import wraps
from inspect import getfullargspec
from copy import deepcopy

from deeptrain.backend import model_utils
from .algorithms import deepmap, obj_to_str
from .experimental import deepcopy_v2
from .configs import PLOT_CFG
from ._backend import WARN, NOTE, TF_KERAS


def pass_on_error(fn, *args, **kwargs):
    errmsg = kwargs.pop('errmsg', None)
    try:
        fn(*args, **kwargs)
    except BaseException as e:
        if errmsg is not None:
            print(errmsg)
        print("Errmsg:", e)


def try_except(try_fn, except_fn):
    try:
        try_fn()
    except:
        if except_fn:  # else pass
            except_fn()


def argspec(obj):
    """Unreliable with wrapped functions."""
    return getfullargspec(obj).args


def _dict_filter_keys(dc, keys, exclude=True, filter_substr=False):
    def condition(k, keys, exclude, filter_substr):
        if not filter_substr:
            value = k in keys
        else:
            value = any([(key in k) for key in keys])
        return (not value) if exclude else value

    keys = keys if isinstance(keys, (list, tuple)) else [keys]
    return {k: v for k, v in dc.items()
            if condition(k, keys, exclude, filter_substr)}


def get_module_methods(module):
    output = {}
    for name in dir(module):
        obj = getattr(module, name)
        obj_name = getattr(obj, '__name__', '')
        if ((str(obj).startswith('<function')
             and isinstance(obj, LambdaType)) # is a function
            and module.__name__ == getattr(obj, '__module__', '')  # same module
            and name in str(getattr(obj, '__code__', ''))  # not a duplicate
            and "__%s__" % obj_name.strip('__') != obj_name  # not a magic method
            and '<lambda>' not in str(getattr(obj, '__code__', ''))  # not lambda
        ):
            output[name] = obj
    return output


def capture_args(fn):
    """Capture bound method arguments without changing its input signature.
    Method must have a `**kwargs` to append captured arguments to.

    Non-literal types and objects will be converted to their string representation
    (or `__qualname__` or `__name__` if they possess it).
    """
    @wraps(fn)
    def wrap(self, *args, **kwargs):
        #### Positional arguments ########
        posarg_names = [arg for arg in argspec(fn)[1:] if arg not in kwargs]
        posargs = {}
        for name, value in zip(posarg_names, args):
            posargs[name] = obj_to_str(value)
        if len(posargs) < len(args):
            varargs = getfullargspec(fn).varargs
            posargs[f'*{varargs}'] = deepmap(args[len(posargs):], obj_to_str)

        #### Keyword arguments ########
        kwargs['_passed_args'] = {}
        if len(kwargs) != 0:
            kwargs['_passed_args'].update(deepcopy_v2(kwargs, obj_to_str))

        kwargs['_passed_args'].update(posargs)
        del kwargs['_passed_args']['_passed_args']
        fn(self, *args, **kwargs)
    return wrap


def _init_optimizer(model, class_weights=None, input_as_labels=False,
                    alias_to_metric_name_fn=None):
    """Instantiates optimizer (and maybe trainer), but does NOT train
    (update weights)."""
    if hasattr(model, '_make_train_function'):
        model._make_train_function()
    else:
        model.optimizer._create_all_weights(model.trainable_weights)


def _make_plot_configs_from_metrics(self):
    """Makes default `plot_configs`, building on `configs._PLOT_CFG`; see
    :meth:`~deeptrain.visuals.get_history_fig`. Validates some configs
    and tries to fill others.

    - Ensures every iterable config is of same `len()` as number of metrics in
      `'metrics'`, by extending *last* value of iterable to match the len. Ex:

      >>> {'metrics': {'val': ['loss', 'accuracy', 'f1']},
      ...  'linestyle': ['--', '-'],  # -> ['--', '-', '-']
      ... }

    - Assigns colors to metrics based on a default cycling coloring scheme,
      with some predefined customs (look for `_customs_map` in source code).
    - Configures up to two plot panes, mediated by `plot_first_pane_max_vals`;
      if number of metrics in `'metrics'` exceeds it, then a second pane is used.
      Can be used to configure how many metrics to draw in first pane; useful
      for managing clutter.
    """
    def _make_colors():
        train_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
        train_defaults.pop(1)  # reserve 'orange' for {'val': 'loss'}
        val_defaults = list(plt.cm.get_cmap('hsv')(np.linspace(.22, 1, 8)))
        train_customs_map = {'loss': train_defaults.pop(0),
                             'accuracy': 'blue'}
        val_customs_map = {'loss': 'orange',
                           'accuracy': 'xkcd:sun yellow',
                           'f1_score': 'purple',
                           'tnr': np.array([0., .503, 1.]),
                           'tpr': 'red'}

        colors = []
        for i, metric in enumerate(self.train_metrics):
            if metric in train_customs_map:
                colors.append(train_customs_map[metric])
            else:
                colors.append(train_defaults[i])

        for metric in self.val_metrics:
            if metric in val_customs_map:
                colors.append(val_customs_map[metric])
            else:
                colors.append(val_defaults.pop(0))
        return colors

    def _get_extend(config, n, tail=False):
        if not isinstance(config, (tuple, list)):
            config = [config]
        cfg = config[n:] if tail else config[:n]
        if len(cfg) < n:
            cfg.extend([cfg[-1]] * (n - len(cfg)))
        return cfg

    n_train = len(self.train_metrics)
    n_val = len(self.val_metrics)

    val_metrics_p1 = self.val_metrics[:self.plot_first_pane_max_vals]
    n_val_p1 = len(val_metrics_p1)
    n_total_p1 = n_train + n_val_p1

    colors = _make_colors()
    mark_best_cfg = {'val': self.key_metric,
                     'max_is_best': self.max_is_best}
    _PLOT_CFG = deepcopy(PLOT_CFG)  # ensure module dict remains unchanged
    CFG = _PLOT_CFG['0']

    plot_configs = {'fig_kw': {'figsize': (12, 7)}}  # plt.subplots() kwargs
    plot_configs['0'] = {
        'metrics':
            CFG['metrics'] or {'train': self.train_metrics,
                               'val'  : val_metrics_p1},
        'x_ticks':
            CFG['x_ticks'] or {'train': ['_train_x_ticks'] * n_train,
                               'val'  : ['_val_train_x_ticks'] * n_val_p1},
        'vhlines'      : CFG['vhlines'],
        'mark_best_cfg': CFG['mark_best_cfg'] or mark_best_cfg,
        'ylims'        : CFG['ylims'],
        'legend_kw'    : CFG['legend_kw'],

        'linewidth': _get_extend(CFG['linewidth'], n_total_p1),
        'linestyle': _get_extend(CFG['linestyle'], n_total_p1),
        'color'    : _get_extend(CFG['color'] or colors, n_total_p1),
    }
    if len(self.val_metrics) <= self.plot_first_pane_max_vals:
        return plot_configs

    #### dedicate separate pane to remainder val_metrics ######################
    n_val_p2 = n_val - n_val_p1

    CFG = _PLOT_CFG['1']
    plot_configs['1'] = {
        'metrics':
            CFG['metrics'] or {'val': self.val_metrics[n_val_p1:]},
        'x_ticks':
            CFG['x_ticks'] or {'val': ['_val_x_ticks'] * n_val_p2},
        'vhlines'      : CFG['vhlines'],
        'mark_best_cfg': CFG['mark_best_cfg'] or mark_best_cfg,
        'ylims'        : CFG['ylims'],
        'legend_kw'    : CFG['legend_kw'],

        'linewidth': _get_extend(CFG['linewidth'], n_val_p2),
        'linestyle': _get_extend(CFG['linestyle'], n_val_p2),
        'color'    : _get_extend(CFG['color'] or colors, n_total_p1, tail=True),
    }
    return plot_configs


def _validate_traingen_configs(self):
    """Ensures various attributes are properly configured, and attempts correction
    where possible.
    """
    def _validate_key_metric_fn():
        if self.key_metric_fn is None and 'predict' in self._eval_fn_name:
            from deeptrain import metrics as metrics_fns
            from .training import _get_api_metric_name

            if not isinstance(self.custom_metrics, dict):
                raise TypeError("`custom_metrics` must be a dict")

            loss_name = model_utils.model_loss_name(self.model)
            if self.key_metric == 'loss' and loss_name in self.custom_metrics:
                self.key_metric_fn = self.custom_metrics[loss_name]
            elif self.key_metric not in self.custom_metrics:
                km_name = _get_api_metric_name(self.key_metric,
                                               self.model.loss,
                                               self._alias_to_metric_name)
                # if None, will catch in `_validate_traingen_configs`
                self.key_metric_fn = getattr(metrics_fns, km_name, None)
            else:
                self.key_metric_fn = self.custom_metrics[self.key_metric]
        elif self.key_metric_fn is not None and 'evaluate' in self._eval_fn_name:
            print(NOTE, "`key_metric_fn` is unsued with 'evaluate' in "
                  "`eval_fn.__name__`")

    def _validate_metrics():
        def _validate(metric, failmsg):
            if metric == 'accuracy':
                return  # converted internally (training.py)
            try:
                # check against alias since converted internally when computing
                getattr(deeptrain.metrics, self._alias_to_metric_name(metric))
            except:
                if (not self.custom_metrics or
                    (self.custom_metrics and metric not in self.custom_metrics)):
                    raise ValueError(failmsg)

        model_metrics = model_utils.get_model_metrics(self.model)

        vm_and_eval = self.val_metrics and 'evaluate' in self._eval_fn_name
        if self.val_metrics is None or '*' in self.val_metrics or vm_and_eval:
            if self.val_metrics is None or vm_and_eval:
                if vm_and_eval:
                    print(WARN, "will override `val_metrics` with model metrics "
                          "for 'evaluate' in `eval_fn.__name__`")
                self.val_metrics = model_metrics.copy()
            elif '*' in self.val_metrics:
                for metric in model_metrics:
                    # insert model metrics at wildcard's index
                    self.val_metrics.insert(self.val_metrics.index('*'), metric)
                self.val_metrics.pop(self.val_metrics.index('*'))

        for name in ('train_metrics', 'val_metrics'):
            value = getattr(self, name)
            if not isinstance(value, list):
                if isinstance(value, (str, type(None))):
                    setattr(self, name, [value])
                else:
                    setattr(self, name, list(value))
            value = getattr(self, name)
            for i, maybe_alias in enumerate(value):
                getattr(self, name)[i] = self._alias_to_metric_name(maybe_alias)
        self.key_metric = self._alias_to_metric_name(self.key_metric)

        if ('evaluate' in self._eval_fn_name and
            self.key_metric not in model_metrics):
            raise ValueError(f"key_metric {self.key_metric} must be in one of "
                             "metrics returned by model, when using 'evaluate' "
                             "in `eval_fn.__name__`. (model returns: %s)"
                             % ', '.join(model_metrics))

        # 'loss' must be in val_metrics, and as first item in list
        if 'loss' not in self.val_metrics:
            self.val_metrics.insert(0, 'loss')
        elif self.val_metrics[0] != 'loss':
            self.val_metrics.pop(self.val_metrics.index('loss'))
            self.val_metrics.insert(0, 'loss')

        loss_name = model_utils.model_loss_name(self.model)
        if (self.key_metric not in self.val_metrics and
            self.key_metric != loss_name):
            self.val_metrics.append(self.key_metric)

        if 'predict' in self._eval_fn_name:
            for metric in self.val_metrics:
                if metric == 'loss':
                    metric = loss_name
                _validate(metric, failmsg=("'{0}' metric is not supported; add "
                                           "a function to `custom_metrics` as "
                                           "'{0}': func.").format(metric))
            _validate(loss_name, failmsg=(
                "'{0}' loss is not supported w/ `eval_fn_name = 'predict'`; "
                "add a function to `custom_metrics` as '{0}': func, or set "
                "`eval_fn_name = 'evaluate'`.").format(loss_name))

        if self.max_is_best and self.key_metric == 'loss':
            print(NOTE + "`max_is_best = True` and `key_metric = 'loss'`"
                  "; will consider higher loss to be better")

    def _validate_directories():
        if self.logs_dir is None and self.best_models_dir is None:
            print(WARN, "`logs_dir = None` and `best_models_dir = None`; "
                  "logging is OFF")
        elif self.logs_dir is None:
            print(NOTE, "`logs_dir = None`; will not checkpoint "
                  "periodically")
        elif self.best_models_dir is None:
            print(NOTE, "`best_models_dir = None`; best models will not "
                  "be checkpointed")

    def _validate_optimizer_saving_configs():
        for name in ('optimizer_save_configs', 'optimizer_load_configs'):
            cfg = getattr(self, name)
            if cfg is not None and 'include' in cfg and 'exclude' in cfg:
                raise ValueError("cannot have both 'include' and 'exclude' "
                                 f"in `{name}`")

    def _validate_saveskip_list():
        if self.input_as_labels and 'labels' not in self.saveskip_list and (
                '{labels}' not in self.saveskip_list):
            print(NOTE, "will exclude `labels` from saving when "
                  "`input_as_labels=True`; to keep 'labels', add '{labels}'"
                  "to `saveskip_list` instead")
            self.saveskip_list.append('labels')

    def _validate_loadskip_list():
        lsl = self.loadskip_list
        if not isinstance(lsl, list) and lsl not in ('auto', 'none', None):
            raise ValueError("`loadskip_list` must be a list, None, 'auto', "
                             "or 'none'")

    def _validate_weighted_slices_range():
        if self.pred_weighted_slices_range is not None:
            if 'predict' not in self._eval_fn_name:
                raise ValueError("`pred_weighted_slices_range` requires "
                                 "'predict' in `eval_fn_name`")
        if (self.pred_weighted_slices_range is not None or
            self.loss_weighted_slices_range is not None):
            if not (self.datagen.slices_per_batch and
                    self.val_datagen.slices_per_batch):
                raise ValueError("to use `loss_weighted_slices_range`, and/or "
                                 "`pred_weighted_slices_range`, "
                                 "`datagen` and `val_datagen` must have "
                                 "`slices_per_batch` attribute set (not falsy) "
                                 "(via `preprocessor`).")

    def _validate_class_weights():
        for name in ('class_weights', 'val_class_weights'):
            cw = getattr(self, name)
            if cw is not None:
                if not all(isinstance(x, int) for x in cw.keys()):
                    raise ValueError(("`{}` classes must be of type int (got {})"
                                      ).format(name, cw))
                if not ((0 in cw and 1 in cw) or sum(cw.values()) > 1):
                    raise ValueError(("`{}` must contain classes 1 and 0, or "
                                     "greater (got {})").format(name, cw))

                if self.model.loss in ('categorical_crossentropy',
                                       'sparse_categorical_crossentropy'):
                    n_classes = self.model.output_shape[-1]
                    for class_label in range(n_classes):
                        if class_label not in cw:
                            getattr(self, name)[name][class_label] = 1

    def _validate_best_subset_size():
        if self.best_subset_size is None:
            return
        elif not isinstance(self.best_subset_size, int):
            raise TypeError("`best_subset_size` must be an int, got %s"
                            % self.best_subset_size)
        elif self.best_subset_size < 1:
            raise ValueError("`best_subset_size` must be >=1 or None, got %s"
                             % self.best_subset_size)

        if self.val_datagen.shuffle_group_samples:
            raise ValueError("`val_datagen` cannot use `shuffle_group_"
                             "samples` with `best_subset_size`, as samples must "
                             "remain in original batches (tracked by respective "
                             '`set_num`s) for resulting "best subset" `set_num`s '
                             "to map to actual samples.")

    def _validate_dynamic_predict_threshold_min_max():
        if self.dynamic_predict_threshold_min_max is None:
            return
        elif 'pred_threshold' not in argspec(self.key_metric_fn):
            raise ValueError("`pred_threshold` parameter missing from "
                             "`key_metric_fn`; cannot use "
                             "`dynamic_predict_threshold_min_max`")

    def _validate_or_make_plot_configs():
        defaults = self._make_plot_configs_from_metrics()
        if self.plot_configs is not None:
            configs = self.plot_configs
            if (not isinstance(configs, dict) or
                not all(isinstance(cfg, dict) for cfg in configs.values())):
                raise TypeError("`plot_configs` must be a dict of dicts, got:"
                                "\n%s" % configs)

            # insert defaults where keys/subkeys absent
            for cfg_name, cfg_default in defaults.items():
                if cfg_name not in configs:
                    configs[cfg_name] = cfg_default.copy()
                else:
                    for key in cfg_default:
                        if key not in configs[cfg_name]:
                            configs[cfg_name][key] = cfg_default[key]
            self.plot_configs = configs
        else:
            self.plot_configs = defaults

    def _validate_metric_printskip_configs():
        for name, cfg in self.metric_printskip_configs.items():
            if not isinstance(cfg, list):
                if isinstance(cfg, tuple):
                    self.metric_printskip_configs[name] = list(cfg)
                else:
                    self.metric_printskip_configs[name] = [cfg]

    def _validate_callbacks():
        def _validate_types(cb, stage):
            if not isinstance(cb[stage], (list, tuple)):
                cb[stage] = (cb[stage],)
            for fn in cb[stage]:
                if not isinstance(fn, LambdaType):
                    raise ValueError("`callbacks` dict values must be "
                                     "functions, or list or tuple of functions.")

        supported = ('save', 'load', 'val_end',
                     'train:iter', 'train:batch', 'train:epoch',
                     'val:iter', 'val:batch', 'val:epoch')
        if not isinstance(self.callbacks, (list, tuple, dict)):
            raise ValueError("`callbacks` must be list, tuple, or dict "
                             "- got %s" % type(self.callbacks))
        if isinstance(self.callbacks, dict):
            self.callbacks = [self.callbacks]

        from ..callbacks import TraingenCallback

        for cb in self.callbacks:
            if not isinstance(cb, (dict, TraingenCallback)):
                raise TypeError("`callbacks` items must be dict or subclass "
                                "TraingenCallback, got %s" % type(cb))
            if isinstance(cb, TraingenCallback):
                continue
            for stage in cb:
                stages = stage if isinstance(stage, tuple) else (stage,)
                if not all(s in supported for s in stages):
                    raise ValueError(f"stage '{stage}' in `callbacks` is not "
                                     "supported; supported are: "
                                     + ', '.join(supported))
                _validate_types(cb, stage)

    def _validate_model_save_kw():
        def _set_save_format(kw):
            save_format = kw.get('save_format', None)
            if save_format is None:
                if TF_KERAS:
                    kw['save_format'] = 'h5'
                else:
                    kw.pop('save_format', None)
            elif not TF_KERAS:
                if save_format == 'h5':  # `keras` saves in 'h5' automatically
                    kw.pop('save_format', None)
                else:
                    # don't raise on 'h5' since it still yields desired behavior
                    raise ValueError("`keras` does not support `save_format` "
                                     "kwarg")

        if self.model_save_kw is None:
            self.model_save_kw = {'include_optimizer': True}
            if TF_KERAS:
                self.model_save_kw['save_format'] = 'h5'
        else:
            _set_save_format(self.model_save_kw)

        if self.model_save_weights_kw is None:
            self.model_save_weights_kw = {'save_format': 'h5'} if TF_KERAS else {}
        else:
            _set_save_format(self.model_save_weights_kw)

    def _validate_freq_configs():
        for name in ('val_freq', 'plot_history_freq', 'unique_checkpoint_freq',
                     'temp_checkpoint_freq'):
            attr = getattr(self, name)
            if not isinstance(attr, (dict, type(None))):
                raise TypeError(f"{name} must be dict or None (got: {attr})")
            elif isinstance(attr, dict) and len(attr) > 1:
                raise ValueError(f"{name} supports up to one key-value pair "
                                 f"(got: {attr})")

    def _validate_model_name_configs():
        if self.model_name_configs.get('best_key_metric', None) is None:
            self.model_name_configs['best_key_metric'
                                    ] = ('__max' if self.max_is_best else '__min')

    def _validate_inputs_as_labels():
        if (not self.input_as_labels and
            (self.datagen.labels_path is None or
             self.val_datagen.labels_path is None)):
            raise Exception("if `input_as_labels=False`, `datagen` and "
                            "`val_datagen` must have `labels_path` defined.")

    loc_names = list(locals())
    for name in loc_names:
        if name.startswith('_validate_'):
            locals()[name]()


def append_examples_dir_to_sys_path():
    """Enables utils.py to be imported for examples."""
    import inspect
    from pathlib import Path
    pkgdir = Path(inspect.stack()[0][1]).parents[2]
    exdir = Path(pkgdir, "examples")
    if not exdir.is_dir():
        raise Exception("`examples` directory isn't on same level as deeptrain "
                        "(%s)" % exdir)

    utilsdir = str(Path(str(exdir), "dir"))
    import sys
    sys.path.insert(0, utilsdir)
    while utilsdir in sys.path[1:]:
        sys.path.pop(sys.path.index(utilsdir))  # avoid duplication
