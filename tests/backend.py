import os
import sys
import contextlib
import shutil
import tempfile
import pytest
import numpy as np
import tensorflow as tf

from pathlib import Path
from termcolor import cprint

#### Environment configs ######################################################
# for testing locally
os.environ['TF_KERAS'] = os.environ.get("TF_KERAS", '1')
os.environ['TF_EAGER'] = os.environ.get("TF_EAGER", '1')

BASEDIR = str(Path(__file__).parents[1])
TF_KERAS = bool(os.environ['TF_KERAS'] == '1')
TF_EAGER = bool(os.environ['TF_EAGER'] == '1')
TF_2 = bool(tf.__version__[0] == '2')

if TF_2:
    USING_GPU = bool(tf.config.list_logical_devices('GPU') != [])
else:
    USING_GPU = bool(tf.config.experimental.list_logical_devices('GPU') != [])

if not TF_EAGER:
    tf.compat.v1.disable_eager_execution()
elif not TF_2:
    raise Exception("deeptrain does not support TF1 in Eager execution")

print(("{}\nTF version: {}\nTF uses {}\nTF executing in {} mode\n"
       "TF_KERAS = {}\n{}\n").format("=" * 80,
                                     tf.__version__,
                                     "GPU"   if USING_GPU else "CPU",
                                     "Eager" if TF_EAGER  else "Graph",
                                     "1"     if TF_KERAS  else "0",
                                     "=" * 80))

pyxfail = pytest.mark.xfail(TF_2 and TF_KERAS and not TF_EAGER,
                            reason="TF2.2 Graph botched `sample_weight`")

#### Imports + Funcs ##########################################################
from deeptrain import util
from deeptrain import metrics
from deeptrain import TrainGenerator, DataGenerator

if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras import losses as keras_losses
    from tensorflow.keras import metrics as keras_metrics
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.layers import Activation, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model, load_model
else:
    from keras import backend as K
    from keras import losses as keras_losses
    from keras import metrics as keras_metrics
    from keras.layers import Input, Dense, LSTM, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from keras.layers import Activation, BatchNormalization
    from keras.regularizers import l2
    from keras.optimizers import Adam
    from keras.models import Model, load_model


def _init_session(C, weights_path=None, loadpath=None, model=None,
                  model_fn=None):
    if model is None:
        model = model_fn(weights_path=weights_path, **C['model'])
    dg  = DataGenerator(**C['datagen'])
    vdg = DataGenerator(**C['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, loadpath=loadpath, **C['traingen'])
    return tg


def _do_test_load(tg, C, init_session_fn):
    def _get_latest_paths(logdir):
        paths = [str(p) for p in Path(logdir).iterdir() if p.suffix == '.h5']
        paths.sort(key=os.path.getmtime)
        return ([p for p in paths if '__weights' in Path(p).stem][-1],
                [p for p in paths if '__state' in Path(p).stem][-1])

    logdir = tg.logdir
    tg.destroy(confirm=True)
    del tg

    weights_path, loadpath = _get_latest_paths(logdir)
    init_session_fn(C, weights_path, loadpath)


@contextlib.contextmanager
def tempdir(dirpath=None):
    if dirpath is not None and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)
    elif dirpath is None:
        dirpath = tempfile.mkdtemp()
    else:
        os.mkdir(dirpath)
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def _get_test_names(module_name):
    module = sys.modules[module_name]
    names = []
    for name in dir(module):
        if name.startswith('test_') or name.startswith('_test_'):
            names.append(name.split('test_')[-1])
    return names


def notify(tests_done):
    def wrap(test_fn):
        def _notify(monkeypatch, *args, **kwargs):
            try:
                is_mp = monkeypatch.__class__.__name__ == 'MonkeyPatch'
            except:
                test_fn(*args, **kwargs)

            if ('monkeypatch' in util.misc.argspec(test_fn) and is_mp
                ) or not is_mp:
                test_fn(monkeypatch, *args, **kwargs)
            elif is_mp:
                test_fn(*args, **kwargs)
            else:
                test_fn(monkeypatch, *args, **kwargs)

            name = test_fn.__name__.split('test_')[-1]
            tests_done[name] = True
            print("\n>>%s TEST PASSED\n" % name.upper())

            if all(tests_done.values()):
                test_name = test_fn.__module__.replace(
                    '_', ' ').replace('tests.', '').upper()
                cprint(f"<< {test_name} PASSED >>\n", 'green')
        return _notify
    return wrap


#### Reusable TrainGenerator, DataGenerator, model configs #####################
batch_size = 128
width, height = 28, 28
channels = 1
datadir = os.path.join(BASEDIR, 'tests', 'data', 'image')

AE_MODEL_CFG = dict(  # autoencoder
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,
    optimizer='nadam',
    activation=['relu'] * 2,
    filters=[2, 1, 2],
    kernel_size=[(3, 3)] * 3,
    strides=[(2, 2), 1, 1],
    up_sampling_2d=[None, (2, 2)],
    input_dropout=.5,
    preout_dropout=.4,
)
CL_MODEL_CFG = dict(  # classifier
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=[8, 16],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=32,
)
IMG_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    superbatch_path=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    shuffle=True,
)
IMG_VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    superbatch_set_nums='all',
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
)
TRAINGEN_CFG = dict(
    epochs=1,
    val_freq={'epoch': 1},
    logs_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_logs'),
    best_models_dir=os.path.join(BASEDIR, 'tests', '_outputs', '_models'),
)
AE_TRAINGEN_CFG = TRAINGEN_CFG.copy()
CL_TRAINGEN_CFG = TRAINGEN_CFG.copy()
AE_TRAINGEN_CFG.update({'model_configs': AE_MODEL_CFG,
                        'input_as_labels': True,
                        'max_is_best': False})
CL_TRAINGEN_CFG.update({'model_configs': CL_MODEL_CFG})

data_cfgs = {'datagen':     IMG_DATAGEN_CFG,
             'val_datagen': IMG_VAL_DATAGEN_CFG}
AE_CONFIGS = {'model':      AE_MODEL_CFG,
              'traingen':   AE_TRAINGEN_CFG}
CL_CONFIGS = {'model':      CL_MODEL_CFG,
              'traingen':   CL_TRAINGEN_CFG}
AE_CONFIGS.update(data_cfgs)
CL_CONFIGS.update(data_cfgs)

#### Model makers #############################################################
def make_classifier(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'metrics', 'optimizer',
                       'num_classes', 'filters', 'kernel_size',
                       'dropout', 'dense_units')
        return [kw[key] for key in expected_kw]

    (batch_shape, loss, metrics, optimizer, num_classes, filters,
     kernel_size, dropout, dense_units) = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = ipt

    for f, ks in zip(filters, kernel_size):
        x = Conv2D(f, ks, activation='relu', padding='same')(x)

    x   = MaxPooling2D(pool_size=(2, 2))(x)
    x   = Dropout(dropout[0])(x)
    x   = Flatten()(x)
    x   = Dense(dense_units, activation='relu')(x)

    x   = Dropout(dropout[1])(x)
    x   = Dense(num_classes)(x)
    out = Activation('softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


def make_autoencoder(batch_shape, optimizer, loss, metrics,
                     filters, kernel_size, strides, activation, up_sampling_2d,
                     input_dropout, preout_dropout, weights_path=None):
    """28x compression, denoising AutoEncoder."""
    ipt = Input(batch_shape=batch_shape)
    x   = ipt
    x   = Dropout(input_dropout)(x)

    configs = (activation, filters, kernel_size, strides, up_sampling_2d)
    for a, f, ks, s, ups in zip(*configs):
        x = UpSampling2D(ups)(x) if ups else x
        x = Conv2D(f, ks, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(a)(x)

    x   = Dropout(preout_dropout)(x)
    x   = Conv2D(1, (3, 3), strides=1, padding='same', activation='sigmoid')(x)
    out = x

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)
    return model


def make_timeseries_classifier(weights_path=None, **kw):
    def _unpack_configs(kw):
        expected_kw = ('batch_shape', 'loss', 'units', 'optimizer')
        return [kw[key] for key in expected_kw]

    batch_shape, loss, units, optimizer = _unpack_configs(kw)

    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, return_sequences=False, stateful=True,
               kernel_regularizer=l2(1e-4),
               recurrent_regularizer=l2(1e-4),
               bias_regularizer=l2(1e-4))(ipt)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)

    if weights_path is not None:
        model.load_weights(weights_path)
    return model


#### Dummy objects ############################################################
class ModelDummy():
    """Proxy model for testing (e.g. methods via `self`)"""
    def __init__(self):
        self.loss = 'mse'
        self.output_shape = (8, 1)
        self.input_shape = (8, 16, 2)

        self._compile_metrics = None

    def _standardize_user_data(self, *args, **kwargs):
        pass

    def _make_train_function(self, *args, **kwargs):
        pass


class TraingenDummy():
    """Proxy class for testing (e.g. methods via `self`)"""

    class Datagen():
        def __init__(self):
            self.shuffle = False

    def __init__(self):
        self.model = ModelDummy()
        self.datagen = TraingenDummy.Datagen()
        self.val_datagen = TraingenDummy.Datagen()

        self._eval_fn_name = 'predict'
        self.key_metric = 'f1_score'
        self.key_metric_fn = metrics.f1_score
        self.max_is_best = True
        self.class_weights = None
        self.val_class_weights = None
        self.batch_size = 8
        self._inferred_batch_size = 8

        self.best_subset_size = None
        self.pred_weighted_slices_range = None
        self.predict_threshold = .5
        self.dynamic_predict_threshold = .5
        self.dynamic_predict_threshold_min_max = None
        self.loss_weighted_slices_range = None
        self.pred_weighted_slices_range = None

        self.train_metrics = []
        self.val_metrics = []
        self._sw_cache = []
        self.custom_metrics = {}

        self.logs_dir = os.path.join(BASEDIR, 'tests', '_outputs', '_logs')
        self.best_models_dir = os.path.join(BASEDIR, 'tests', '_outputs',
                                            '_models')
        self.model_configs = None
        self.model_name_configs = {}
        self.new_model_num = False
        self.model_base_name = 'M'
        self.name_process_key_fn = util.configs.NAME_PROCESS_KEY_FN
        self.alias_to_metric = util._default_configs._DEFAULT_ALIAS_TO_METRIC

        self.optimizer_save_configs = None
        self.optimizer_load_configs = None
        self.metric_printskip_configs = {}
        self.model_save_kw = None
        self.model_save_weights_kw = None
        self.val_freq = {'epoch': 1}
        self.plot_history_freq = {'epoch': 1}
        self.unique_checkpoint_freq = {'epoch': 1}
        self.temp_checkpoint_freq = {'epoch': 1}
        self.callbacks = {}

        self.input_as_labels = False
        self.loadskip_list = None
        self.saveskip_list = None
        self.plot_configs = None
        self.plot_first_pane_max_vals = 2

    def set_shapes(self, batch_size, label_dim):
        self.batch_size = batch_size
        self._inferred_batch_size = batch_size
        self.model.output_shape = (batch_size, label_dim)

    def set_cache(self, y_true, y_pred):
        self._labels_cache = y_true.copy()
        self._preds_cache = y_pred.copy()
        self._sw_cache = np.ones(y_true.shape)
        self._class_labels_cache = y_true.copy()

    def _alias_to_metric_name(self, alias):
        return self.alias_to_metric.get(alias.lower(), alias)


for name in ('_transform_eval_data', '_validate_data_shapes',
             '_validate_class_data_shapes', '_compute_metric', '_compute_metrics',
             '_set_predict_threshold', '_weighted_normalize_preds'):
    setattr(TraingenDummy, name, getattr(util.training, name))
