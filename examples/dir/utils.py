# -*- coding: utf-8 -*-
import os
thisdir = os.path.dirname(__file__)

if os.environ.get('TF_KERAS', '1') == '1':
    from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout
    from tensorflow.keras.layers import BatchNormalization, Activation
    from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
else:
    from keras.layers import Input, Conv2D, UpSampling2D, Dropout
    from keras.layers import BatchNormalization, Activation
    from keras.layers import Dense, MaxPooling2D, Flatten
    from keras.layers import LSTM
    from keras.optimizers import Adam
    from keras.models import Model
    from keras import backend as K
from deeptrain import TrainGenerator, DataGenerator
from deeptrain.callbacks import infer_train_hist_cb
from deeptrain.callbacks import binary_preds_per_iteration_cb
from deeptrain.callbacks import binary_preds_distribution_cb


def init_session(CONFIGS, model_fn):
    model = model_fn(**CONFIGS['model'])
    dg  = DataGenerator(**CONFIGS['datagen'])
    vdg = DataGenerator(**CONFIGS['val_datagen'])
    tg  = TrainGenerator(model, dg, vdg, **CONFIGS['traingen'])
    return tg

#### Reusable TrainGenerator, DataGenerator, model configs ####################
batch_size = 128
width, height, channels = 28, 28, 1
datadir = os.path.join(thisdir, 'data', 'image')

DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    batch_size=batch_size,
    shuffle=True,
    superbatch_set_nums='all',
)
VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    batch_size=batch_size,
    shuffle=False,
    superbatch_set_nums='all',
)
TRAINGEN_CFG = dict(
    epochs=6,
    logs_dir=os.path.join(thisdir, 'logs'),
    best_models_dir=os.path.join(thisdir, 'models'),
    eval_fn='predict',
)
#### Reusable AutoEncoder #####################################################
def make_autoencoder(batch_shape, optimizer, loss, metrics,
                     filters, kernel_size, strides, activation, up_sampling_2d,
                     input_dropout, preout_dropout, lr=None):
    ipt = Input(batch_shape=batch_shape)
    x   = Dropout(input_dropout)(ipt)

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
    if lr:
        K.set_value(model.optimizer.learning_rate, lr)
    return model

AE_MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='mse',
    metrics=None,
    optimizer='nadam',
    activation=['relu'] * 5,
    filters=[6, 12, 2, 6, 12],
    kernel_size=[(3, 3)] * 5,
    strides=[(2, 2), (2, 2), 1, 1, 1],
    up_sampling_2d=[None, None, None, (2, 2), (2, 2)],
    input_dropout=.5,
    preout_dropout=.4,
)
AE_TRAINGEN_CFG = TRAINGEN_CFG.copy()
AE_TRAINGEN_CFG.update({'model_configs': AE_MODEL_CFG,
                        'input_as_labels': True,
                        'max_is_best': False})
AE_CONFIGS = {'model':       AE_MODEL_CFG,
              'datagen':     DATAGEN_CFG,
              'val_datagen': VAL_DATAGEN_CFG,
              'traingen':    AE_TRAINGEN_CFG}

#### Reusable Classifier ####################################################
def make_classifier(batch_shape, loss, metrics, optimizer, num_classes,
                    filters, kernel_size, dropout, dense_units):
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
    return model

CL_MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(4e-4),
    num_classes=10,
    filters=[8, 16],
    kernel_size=[(3, 3), (3, 3)],
    dropout=[.25, .5],
    dense_units=32,
)
img_labels_paths = [os.path.join(datadir, "train", "labels.h5"),
                    os.path.join(datadir, "val",   "labels.h5")]

CL_DATAGEN_CFG = DATAGEN_CFG.copy()
CL_VAL_DATAGEN_CFG = VAL_DATAGEN_CFG.copy()
CL_DATAGEN_CFG['labels_path'] = img_labels_paths[0]
CL_VAL_DATAGEN_CFG['labels_path'] = img_labels_paths[1]

CL_TRAINGEN_CFG = TRAINGEN_CFG.copy()
CL_TRAINGEN_CFG.update({'model_configs': CL_MODEL_CFG})
CL_CONFIGS = {'model':       CL_MODEL_CFG,
              'datagen':     CL_DATAGEN_CFG,
              'val_datagen': CL_VAL_DATAGEN_CFG,
              'traingen':    CL_TRAINGEN_CFG}

#### Reusable Timeseries Classifier ##########################################
def make_timeseries_classifier(batch_shape, loss, optimizer, units, activation):
    ipt = Input(batch_shape=batch_shape)
    x   = LSTM(units, activation=activation, return_sequences=True,
               stateful=True)(ipt)
    x   = LSTM(units, activation=activation, return_sequences=False,
               stateful=True)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss)
    return model

ts_datadir = os.path.join(thisdir, 'data', 'ptbdb')
ts_batch_size = 128
window_size = ts_batch_size / 4.
assert window_size.is_integer()
window_size = int(window_size)

TS_MODEL_CFG = dict(
    batch_shape=(ts_batch_size, window_size, 1),
    units=24,
    optimizer='adam',
    loss='binary_crossentropy',
    activation='tanh',
)
TS_DATAGEN_CFG = dict(
    data_path=os.path.join(ts_datadir, 'train', 'data.h5'),
    labels_path=os.path.join(ts_datadir, 'train', 'labels.h5'),
    batch_size=ts_batch_size,
    superbatch_set_nums='all',
    shuffle=True,
    preprocessor='timeseries',
    preprocessor_configs=dict(window_size=window_size),
)
TS_VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(ts_datadir, 'val', 'data.h5'),
    labels_path=os.path.join(ts_datadir, 'val', 'labels.h5'),
    batch_size=ts_batch_size,
    superbatch_set_nums='all',
    shuffle=False,
    preprocessor='timeseries',
    preprocessor_configs=dict(window_size=window_size),
)
TS_TRAINGEN_CFG = dict(
    epochs=4,
    reset_statefuls=True,
    logs_dir=os.path.join(thisdir, 'logs'),
    best_models_dir=os.path.join(thisdir, 'models'),
    model_configs=TS_MODEL_CFG,
    callbacks={'val_end': [infer_train_hist_cb,
                           binary_preds_per_iteration_cb,
                           binary_preds_distribution_cb]},
    val_freq={'epoch': 2},
    plot_history_freq={'epoch': 2},
    unique_checkpoint_freq={'epoch': 2},
)
TS_CONFIGS = {'model': TS_MODEL_CFG, 'datagen': TS_DATAGEN_CFG,
              'val_datagen': TS_VAL_DATAGEN_CFG, 'traingen': TS_TRAINGEN_CFG}
