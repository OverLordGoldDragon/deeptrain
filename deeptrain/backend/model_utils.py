from types import LambdaType
from . import TF_KERAS, tf_eager, TF_2


def get_model_metrics(model):
    # TF1, 2, Eager, Graph, keras, and tf.keras store model.compile(metrics)
    # differently
    if TF_2 and TF_KERAS:
        if tf_eager():
            metrics = model.compiled_metrics._user_metrics
        else:
            metrics = model._compile_metrics
    else:
        metrics = model.metrics_names

    if metrics and 'loss' in metrics:
        metrics.pop(metrics.index('loss'))
    return ['loss', *metrics] if metrics else ['loss']


def model_loss_name(model):
    if not hasattr(model, 'loss'):
        raise AttributeError("`model` has no attribute 'loss'; did you run "
                             "`model.compile()`?")
    loss = model.loss
    if isinstance(loss, str):
        return loss
    elif isinstance(loss, LambdaType):
        return loss.__name__
    else:
        raise Exception("unable to get name of `model` loss")
