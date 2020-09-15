# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
   - Inspecting gradients per layer
   - Estimating good values of gradient clipping threshold
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()
from utils import make_autoencoder, init_session
from utils import AE_CONFIGS as C

import numpy as np
from tensorflow.keras.optimizers import Adam

#%%# Configure training #######################################################
tg = init_session(C, make_autoencoder)

#%%# Expected gradient norm estimation ########################################
# We iterate over entire train dataset, gathering gradients from every fit
# and computing and storing their L2-norms.
grad_norms, *_ = tg.gradient_norm_over_dataset()

#%%# We can now restart training with setting optimizer `clipnorm` to 1.5x average
# value, avoiding extreme gradients while not clipping most standard gradients
C['model']['optimizer'] = Adam(clipnorm=1.5 * np.mean(grad_norms))
tg = init_session(C, make_autoencoder)
tg.epochs = 1  # train just for demo
tg.train()

#%%# Complete gradident sum ###################################################
# This time we run a cumulative sum over actual gradient tensors, preserving
# and returning their shapes, allowing per-weight visualization
plot_kw = {'h': 2}  # double default height since we expect many weights
grads_sum, *_ = tg.gradient_sum_over_dataset(plot_kw=plot_kw)

# We can use the mean of `grads_sum` to set `clipvalue` instead of `clipnorm`
