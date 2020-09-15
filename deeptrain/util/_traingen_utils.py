from deeptrain.util import logging, saving, searching, training, misc
from deeptrain import introspection, visuals
from .misc import argspec


class TraingenUtils():
    # define explicitly to enable indexing (linter, docs)

    checkpoint = saving.checkpoint
    save       = saving.save
    load       = saving.load
    _save_best_model      = saving._save_best_model
    _make_model_save_fns  = saving._make_model_save_fns
    _get_optimizer_state  = saving._get_optimizer_state
    _load_optimizer_state = saving._load_optimizer_state
    _save_history_fig     = saving._save_history_fig

    save_report           = logging.save_report
    generate_report       = logging.generate_report
    get_unique_model_name = logging.get_unique_model_name
    get_last_log          = logging.get_last_log

    _update_temp_history         = training._update_temp_history
    get_sample_weight            = training.get_sample_weight
    _get_weighted_sample_weight  = training._get_weighted_sample_weight
    _set_predict_threshold       = training._set_predict_threshold
    _get_val_history             = training._get_val_history
    _get_best_subset_val_history = training._get_best_subset_val_history
    _compute_metric              = training._compute_metric
    _compute_metrics             = training._compute_metrics
    _transform_eval_data         = training._transform_eval_data
    _weighted_normalize_preds    = training._weighted_normalize_preds
    _validate_data_shapes        = training._validate_data_shapes
    _validate_class_data_shapes  = training._validate_class_data_shapes

    get_history_fig = visuals.get_history_fig

    compute_gradient_norm      = introspection.compute_gradient_norm
    gradient_norm_over_dataset = introspection.gradient_norm_over_dataset
    gradient_sum_over_dataset  = introspection.gradient_sum_over_dataset
    _gather_over_dataset       = introspection._gather_over_dataset
    interrupt_status           = introspection.interrupt_status
    info                       = introspection.info

    _make_plot_configs_from_metrics = misc._make_plot_configs_from_metrics
    _validate_traingen_configs      = misc._validate_traingen_configs

    def __init__(self):
        pass

# valiate that all pertinent methods have been explicitly bounded
modules = (logging, saving, searching, training, misc, introspection, visuals)
to_exclude = ['_log_init_state', '_update_best_key_metric_in_model_name']

for module in modules:
    mm = misc.get_module_methods(module)
    for name, method in mm.items():
        if name in to_exclude or 'self' not in argspec(method):
            continue
        if not hasattr(TraingenUtils, name):
            raise AttributeError(f"TraingenUtils is missing a method: {name}")
        elif getattr(TraingenUtils, name) is not method:
            raise AttributeError(("TraingenUtils.{} should be {} (got {})"
                                  ).format(name, method,
                                           getattr(TraingenUtils, name)))
for name in to_exclude:
    if hasattr(TraingenUtils, name):
        raise AttributeError(f"'{name}' shouldn't be set for TraingenUtils")
