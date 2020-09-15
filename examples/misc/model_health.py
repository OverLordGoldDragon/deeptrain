# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - Exploding & vanishing gradients monitoring
    - Spotting dead weights
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import CL_CONFIGS as C
from utils import init_session, make_classifier
from utils import Adam
from see_rnn import rnn_histogram, rnn_heatmap

#%%# Case 1 ###################################################################
# We build a model prone to large but not exploding/vanishing gradients
C['model']['optimizer'] = Adam(10)
C['traingen']['epochs'] = 1
tg = init_session(C, make_classifier)

#%%# Train ####################################################################
tg.train()

#%%# Case 2 ###################################################################
# Now a model prone to exploding / vanishing gradients
from utils import TS_CONFIGS as C
from utils import make_timeseries_classifier

C['model']['activation'] = 'relu'
C['model']['optimizer'] = Adam(.3)
C['traingen']['epochs'] = 1
C['traingen']['eval_fn'] = 'predict'
C['traingen']['val_freq'] = {'epoch': 1}
tg = init_session(C, make_timeseries_classifier)

#%%# Train ####################################################################
tg.train()
#%%# Visualize ############################
rnn_histogram(tg.model, 1)
rnn_heatmap(tg.model, 1)
rnn_histogram(tg.model, 2)
rnn_heatmap(tg.model, 2)
