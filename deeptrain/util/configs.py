# -*- coding: utf-8 -*-
"""Custom configurations for :class:`TrainGenerator` / :class:`DataGenerator`
go here. These will be defaulted to when pertinent configs aren't passed
explicitly to `__init__` (e.g. `_PLOT_CFG` for `TrainGenerator.plot_configs`).
"""
import numpy as np
from .fonts import fontsdir
from .algorithms import builtin_or_npscalar


PLOT_CFG = {
'fig_kw': {'figsize': (12, 7)},
'0': {
 'metrics': None,
 'x_ticks': None,
 'vhlines'   :
     {'v': '_hist_vlines',
      'h': 1},
 'mark_best_cfg': None,
 'ylims'        : (0, 2),
 'legend_kw'    : {'fontsize': 13},

 'linewidth': [1.5, 1.5],
 'linestyle': ['-', '-'],
 'color'    : None,
},
'1': {
 'metrics': None,
 'x_ticks': None,
 'vhlines':
     {'v': '_val_hist_vlines',
      'h': .5},
 'mark_best_cfg': None,
 'ylims'        : (0, 1),
 'legend_kw'    : {'fontsize': 13},

 'linewidth': [1.5],
 'linestyle': ['-'],
 'color': None,
}
}


# order-dependent
MODEL_NAME_CFG = dict(
    optimizer       = '',
    lr              = '',
    best_key_metric = None,
)


# * == wildcard (match as substring)
REPORT_CFG = {
    'model':
        {},
    'traingen':
        {
        'exclude':
            ['model', 'model_configs', 'logs_use_full_model_name',
             'history_fig', 'plot_configs', 'max_checkpoints',
             'history', 'val_history', 'temp_history', 'val_temp_history',
             'name_process_key_fn', 'report_fontpath', 'model_name_configs',
             'report_configs', 'datagen', 'val_datagen', 'logdir', 'logs_dir',
             'best_models_dir', 'fit_fn', 'eval_fn', '_fit_fn', '_eval_fn',
             'callbacks', '_cb_alias', '_passed_args', '_history_fig',
             '_metrics_cached', 'metric_printskip_configs',
             '_inferred_batch_size', 'plot_first_pane_max_vals', '_imports',
             'iter_verbosity', '_max_set_name_chars', '_val_max_set_name_chars',
             'metric_to_alias', 'alias_to_metric',
             '*_has_', '*temp_history_empty',
             ],
        'exclude_types':
            [list, np.ndarray, '#best_subset_nums'],
        },
    ('datagen', 'val_datagen'):
        {
        'exclude':
            ['batch', 'group_batch', 'labels', 'all_labels',
             'batch_loaded', 'batch_exhausted', 'set_num', 'set_name',
             '_set_names', 'set_nums_original', 'set_nums_to_process',
             'superbatch_set_nums', 'data_loader', 'data_path',
             'labels_loader', 'labels_path',
             'saveskip_list', 'loadskip_list', '_path_attrs', 'preprocessor',
             '*_ATTRS', '*superbatch', '*_filepaths', '*_filenames']
        },
}


TRAINGEN_SAVESKIP_LIST = [
    'model',
    'optimizer_state',
    'callbacks',
    'key_metric_fn',
    'custom_metrics',
    'metric_to_alias',
    'alias_to_metric',
    'name_process_key_fn',
    '_fit_fn',  # 'fit_fn' & other properties don't show up in `vars(self)`
    '_eval_fn',

    '_labels',
    '_preds',
    '_y_true',
    '_y_preds',
    '_labels_cache',
    '_preds_cache',
    '_sw_cache',

    '_imports',
    '_history_fig',
    '_val_max_set_name_chars',
    '_max_set_name_chars',
    '_inferred_batch_size',
    '_class_labels_cache',
    '_temp_history_empty',
    '_val_temp_history_empty',
    '_val_sw',
    '_set_num',
    '_val_set_num',
]

TRAINGEN_LOADSKIP_LIST = ['{auto}', 'model_name', 'model_base_name',
                          'model_num', 'use_passed_dirs_over_loaded',
                          'logdir', '_init_callbacks_called']

DATAGEN_SAVESKIP_LIST = ['batch', 'superbatch', 'labels', 'all_labels',
                         '_group_batch', '_group_labels']
DATAGEN_LOADSKIP_LIST = ['data_path', 'labels_path', 'superbatch_path',
                         'data_loader', 'set_nums_original',
                         'set_nums_to_process', 'superbatch_set_nums']

MODEL_SAVE_KW = {'include_optimizer': True}
MODEL_SAVE_WEIGHTS_KW = {'save_format': 'h5'}

METRIC_PRINTSKIP_CFG = {
    'train': [],
    'val': [],
}

METRIC_TO_ALIAS = {
    'loss'    : 'Loss',
    'accuracy': 'Acc',
    'f1_score': 'F1',
    'tnr'     : '0-Acc',
    'tpr'     : '1-Acc',
    'mean_absolute_error': 'MAE',
    'mean_squared_error' : 'MSE',
}

ALIAS_TO_METRIC = {
    'acc':     'accuracy',
    'mae':     'mean_absolute_error',
    'mse':     'mean_squared_error',
    'mape':    'mean_absolute_percentage_error',
    'msle':    'mean_squared_logarithmic_error',
    'kld':     'kullback_leibler_divergence',
    'cosine':  'cosine_similarity',
    'f1':      'f1_score',
    'f1-score':'f1_score',
}

def NAME_PROCESS_KEY_FN(key, alias, attrs):
    def _format_float(val, small_th=1e-2):
        def _format_small_float(val):
            def _decimal_len(val):
                return len(val.split('.')[1].split('e')[0])

            val = ('%.3e' % val).replace('-0', '-')
            while '0e' in val:
                val = val.replace('0e', 'e')
            if _decimal_len(val) == 0:
                val = val.replace('.', '')
            return val

        if abs(val) < small_th:
            return _format_small_float(val)
        elif small_th < abs(val) < 1:
            return ("%.3f" % val).lstrip('0').rstrip('0')
        else:
            return ("%.3f" % val).rstrip('0')

    def _squash_list(ls):
        def _max_reps_from_beginning(ls, reps=1):
            if reps < len(ls) and ls[reps] == ls[0]:
                reps = _max_reps_from_beginning(ls, reps + 1)
            return reps

        _str = ''
        while len(ls) != 0:
            reps = _max_reps_from_beginning(ls)
            if isinstance(ls[0], float):
                val = _format_float(ls[0])
            else:
                val = str(ls[0])
            if reps > 1:
                _str += "{}x{}_".format(val, reps)
            else:
                _str += val + '_'
            ls = ls[reps:]
        return _str.rstrip('_')

    def _process_special_keys(key, val):
        if key == 'best_key_metric':
            val = ("%.3f" % val).lstrip('0')
        elif key == 'name':
            val = ''
        elif key == 'timesteps':
            val = val // 1000 if (val / 1000).is_integer() else val / 1000
            val = str(val) + 'k'
        return val

    def _process_val(key, val):
        if not builtin_or_npscalar(val, include_type_type=False):
            if not (hasattr(val, '__name__') or hasattr(type(val), '__name__')):
                raise TypeError(
                    f"cannot encode {val} for model name; `model_configs` values "
                    "must be either Python literals (str, int, etc), or objects "
                    "(or their classes) with  '__name__' attribute. "
                    "Alternatively, set custom `name_process_key_fn`")
            val = val.__name__ if hasattr(val, '__name__') else type(val).__name__
            val = val.split('.')[-1]  # drop packages/modules
        else:
            val = _process_special_keys(key, val)

            if isinstance(val, (list, tuple)):
                val = list(val)  # in case tuple
                val = _squash_list(val)
            if isinstance(val, float):
                val = _format_float(val)
        return val

    val = attrs[key]
    val = _process_val(key, val)

    name = alias if alias is not None else key
    if key != 'best_key_metric':
        return "-{}{}".format(name, val)
    else:
        return "{}{}".format(name, val)


_TRAINGEN_CFG = dict(
    dynamic_predict_threshold_min_max = None,
    checkpoints_overwrite_duplicates  = True,
    loss_weighted_slices_range  = None,
    pred_weighted_slices_range  = None,
    logs_use_full_model_name    = True,
    new_model_num = True,
    dynamic_predict_threshold   = .5,  # initial
    plot_first_pane_max_vals    = 2,
    _val_max_set_name_chars     = 2,
    _max_set_name_chars  = 3,
    predict_threshold    = 0.5,
    best_subset_size     = None,
    check_model_health   = True,
    max_one_best_save    = True,
    max_checkpoints = 5,
    report_fontpath = fontsdir + "consola.ttf",
    model_base_name = "model",
    final_fig_dir   = None,

    loadskip_list = TRAINGEN_LOADSKIP_LIST,
    saveskip_list = TRAINGEN_SAVESKIP_LIST,
    model_save_kw = MODEL_SAVE_KW,
    model_save_weights_kw = MODEL_SAVE_WEIGHTS_KW,
    metric_to_alias       = METRIC_TO_ALIAS,
    alias_to_metric       = ALIAS_TO_METRIC,
    report_configs        = REPORT_CFG,
    model_name_configs    = MODEL_NAME_CFG,
    name_process_key_fn   = NAME_PROCESS_KEY_FN,
    metric_printskip_configs = METRIC_PRINTSKIP_CFG,
)

_DATAGEN_CFG = dict(
    shuffle_group_batches=False,
    shuffle_group_samples=False,
    data_batch_shape=None,
    labels_batch_shape=None,
    data_dtype=None,
    labels_dtype=None,
    loadskip_list=DATAGEN_LOADSKIP_LIST,
    saveskip_list=DATAGEN_SAVESKIP_LIST,
)
