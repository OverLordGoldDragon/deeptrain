# -*- coding: utf-8 -*-
"""The minimal essentials for DeepTrain are:
    - compiled model
    - data directory
This example covers these and a bit more to keep truer to standard use.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.models import Model
from deeptrain import TrainGenerator, DataGenerator

#%%# Configuration ###########################################################
# Begin by defining a model maker function.
# Input should specify hyperparameters, optimizer, learning rate, etc.;
# this is the 'blueprint' which is later saved.
def make_model(batch_shape, optimizer, loss, metrics, num_classes,
               filters, kernel_size):
    ipt = Input(batch_shape=batch_shape)

    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(ipt)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)

    out = Activation('softmax')(x)

    model = Model(ipt, out)
    model.compile(optimizer, loss, metrics=metrics)
    return model

# Set batch_size and specify MNIST dims (28 x 28 pixel, greyscale)
batch_size = 128
width, height, channels = 28, 28, 1

# Define configs dictionary to feed as **kwargs to `make_model`;
# we'll also pass it to TrainGenerator, which will save it and show in a
# "report" for easy reference
MODEL_CFG = dict(
    batch_shape=(batch_size, width, height, channels),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    num_classes=10,
    filters=16,
    kernel_size=(3, 3),
    # `activation`, `pool_size`, & others could be set the same way;
    # good idea if we ever plan on changing them.
)
# Configs for (train) DataGenerator
#   data_path:    directory where image data is located
#   labels_path: where labels file is located
#   batch_size:  number of samples to feed at once to model
#   shuffle:     whether to shuffle data at end of each epoch
#   superbatch_set_nums: which files to load into a `superbatch`, which holds
#       batches persisently in memory (as opposed to `batch`, which is
#       overwritten after use). Since MNIST is small, we can load it all into RAM.
datadir = os.path.join("dir", "data", "image")
DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'train'),
    labels_path=os.path.join(datadir, 'train', 'labels.h5'),
    batch_size=batch_size,
    shuffle=True,
    superbatch_set_nums='all',
)
# Configs for (validation) DataGenerator
VAL_DATAGEN_CFG = dict(
    data_path=os.path.join(datadir, 'val'),
    labels_path=os.path.join(datadir, 'val', 'labels.h5'),
    batch_size=batch_size,
    shuffle=False,
    superbatch_set_nums='all',
)
# Configs for TrainGenerator
#   epochs:   number of epochs to train for
#   logs_dir: where to save TrainGenerator state, model, report, and history
#   best_models_dir: where to save model when it achieves new best
#       validation performance
#   model_configs: model configurations dict to save & write to report
TRAINGEN_CFG = dict(
    epochs=3,
    logs_dir=os.path.join('dir', 'logs'),
    best_models_dir=os.path.join('dir', 'models'),
    model_configs=MODEL_CFG,
)
#%%# Create training objects ################################################
model       = make_model(**MODEL_CFG)
datagen     = DataGenerator(**DATAGEN_CFG)
val_datagen = DataGenerator(**VAL_DATAGEN_CFG)
traingen    = TrainGenerator(model, datagen, val_datagen, **TRAINGEN_CFG)

traingen.epochs = 1
traingen.unique_checkpoint_freq = {'epoch': 2}
traingen.temp_checkpoint_freq = {'epoch': 2}
#%%# Train ##################################################################
traingen.train()
