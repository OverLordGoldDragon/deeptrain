# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - Timeseries binary classification on real data
    - Windowed data format; sequence length 188, 4 windows -> 47 points per window
    - Binary classification visuals
    - Using class weights to handle imbalance
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import TS_CONFIGS as C
from utils import init_session, make_timeseries_classifier
from see_rnn import features_1D, rnn_histogram, rnn_heatmap

#%%# Dataset info #############################################################
# PTB Diagnostic ECG Database - https://www.kaggle.com/shayanfazeli/heartbeat
# Number of samples: 14552
# Number of channels: 1
# Number of classes: 2 (binary classification)
# Sampling frequency: 125 Hz
# Datapoints per sequence: 188
#%%# Configure TrainGenerator, DataGenerators, & model ########################
batch_size = 128
window_size = 188 / 4.    # use 4 windows
assert window_size.is_integer()  # ensure it divides batch_size
window_size = int(window_size)

# Make DataGenerator divide up the (128, 188, 1)-shaped batch
# into 4 slices shaped (128, 47, 1) each, feeding one at a time to model
C['datagen'    ]['preprocessor'] = 'timeseries'
C['val_datagen']['preprocessor'] = 'timeseries'
C['datagen'    ]['preprocessor_configs'] = {'window_size': window_size}
C['val_datagen']['preprocessor_configs'] = {'window_size': window_size}

C['model']['batch_shape'] = (batch_size, window_size, 1)

# eval_fn: need 'predict' for visuals and custom metrics
# key_metric: f1_score for imbalanced binary classification
# val_metrics: true positive rate & true negative rate are "class accuracies",
#              i.e. class-1 acc & class-2 acc
# plot_first_pane_max_vals: plot only validation loss in first plot window,
# the rest on second, to avoid clutter and keep losses together
# class_weights: "normal" is the minority class; 3x more "abnormal" samples
# others: see utils.py
C['traingen'].update(dict(
    eval_fn='predict',
    key_metric='f1_score',
    val_metrics=('loss', 'tnr', 'tpr'),
    plot_first_pane_max_vals=1,
    class_weights={0: 3, 1: 1},
))
tg = init_session(C, make_timeseries_classifier)
#%%# Visualize 24 samples #####################################################
data = tg.val_datagen.batch
features_1D(data[:24], n_rows=6, subplot_samples=True, tight=True)
#%%# Train ####################################################################
tg.train()
#%%# Visualize LSTM weights post-training #####################################
rnn_heatmap(tg.model, 1)  # 1 == layer index
rnn_histogram(tg.model, 1)
