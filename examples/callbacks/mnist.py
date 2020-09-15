# -*- coding: utf-8 -*-
"""This example assumes you've read `callbacks/basic.py`, and covers:
    - Creating advanced custom callbacks
    - Using and modifying builtin callbacks
    - Visualization, data gathering, and random seed setting callbacks
"""
import os
import sys
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()
logger_savedir = os.path.join(sys.path[0], "logger")

from utils import make_classifier, init_session, img_labels_paths
from utils import CL_CONFIGS as C
from see_rnn import features_2D
import numpy as np

from deeptrain.callbacks import TraingenCallback, TraingenLogger
from deeptrain.callbacks import RandomSeedSetter
from deeptrain.callbacks import make_layer_hists_cb

#%%#
# TraingenLogger gathers data throughout training: weights, outputs, and
# gradients of model layers. We inherit the base class and override
# methods where we wish actions to occur: on save, load, and end of train epoch.
class TraingenLoggerCB(TraingenLogger):
    def __init__(self, savedir, configs, **kwargs):
        super().__init__(savedir, configs, **kwargs)

    def on_save(self, stage=None):
        self.save(_id=self.tg.epoch)  # `tg` will be set inside TrainGenerator

    def on_load(self, stage=None):
        self.clear()
        self.load()

    def on_train_epoch_end(self, stage=None):
        self.log()

log_configs = {
    'weights': ['conv2d'],
    'outputs': 'conv2d',
    'gradients': ('conv2d',),
    'outputs-kw': dict(learning_phase=0),
    'gradients-kw': dict(learning_phase=0),
}
tglogger = TraingenLoggerCB(logger_savedir, log_configs)

#%%#
# Plots model outputs in a heatmap at end of every 2 epochs.
# Relies on `TraingenLogger` being included in `callbacks`, which stores
# model outputs so they aren't recomputed for visualization.
# All callback objects (except funcs in dicts) are required to subclass
# TraingenCallback (TraingenLogger does so)
class OutputsHeatmap(TraingenCallback):
    def on_val_end(self, stage=None):
        if stage == ('val_end', 'train:epoch') and (self.tg.epoch % 2) == 0:
            # run `viz` within `TrainGenerator._on_val_end`,
            # and on every other epoch
            self.viz()

    def viz(self):
        data = self._get_data()
        features_2D(data, tight=True, title=False, cmap='hot',
                    norm=None, show_xy_ticks=[0, 0], w=1.1, h=.55, n_rows=4)

    def _get_data(self):
        lg = None
        for cb in self.tg.callbacks:
            if isinstance(cb, TraingenLogger):
                lg = cb
        if lg is None:
            raise Exception("TraingenLogger not found in `callbacks`")

        last_key = list(lg.outputs.keys())[-1]
        outs = list(lg.outputs[last_key][0].values())[0]
        sample = outs[0]                  # (width, height, channels)
        return sample.transpose(2, 0, 1)  # (channels, width, height)

outputs_heatmap = OutputsHeatmap()

#%%#
# Plots weights of the second Conv2D layer at end of each epoch.
# Weights are reshaped such that subplot 'boxes' are output channels, and each
# box plots flattened spatial dims vertically and input features horizontally.
class ConvWeightsHeatmap(TraingenCallback):
    def on_val_end(self, stage=None):
        if stage == ('val_end', 'train:epoch'):
            self.viz()

    def viz(self):
        w = self.tg.model.layers[2].get_weights()[0]
        w = w.reshape(-1, *w.shape[2:])  # flatten along spatial dims
        w = w.transpose(2, 0, 1)  # (out_features, spatial dims x in_features)

        if not hasattr(self, 'init_norm'):
            # maintain same norm throughout plots for consistency
            mx = np.max(np.abs(w))
            self.init_norm = (-mx, mx)

        features_2D(w, tight=True, w=.4, h=.4, title=None, show_xy_ticks=0,
                    norm=self.init_norm)

cwh = ConvWeightsHeatmap()
#%%
# Callbacks can also be configured as str-function dict pairs, where str
# is name of a callback "stage" (see tg._cb_alias after tg.train()).
grad_hists = {'train:epoch': [make_layer_hists_cb(mode='gradients:outputs'),
                              make_layer_hists_cb(mode='gradients:weights')]}
weight_hists = {('val_end', 'train:epoch'): make_layer_hists_cb(mode='weights')}

configs = {'title': dict(fontsize=13), 'plot': dict(annot_kw=None)}
layer_outputs_hists = {'val_end':
                       make_layer_hists_cb(mode='outputs', configs=configs)}
#%%#
# Set new random seeds (`random`, `numpy`, TF-graph, TF-global) every epoch,
# incrementing by 1 from start value (default 0)
seed_setter = RandomSeedSetter(freq={'train:epoch': 2})
#%%###########################################################################
C['traingen']['callbacks'] = [seed_setter, tglogger, outputs_heatmap, cwh,
                              grad_hists, weight_hists, layer_outputs_hists]
C['traingen']['epochs'] = 16
C['datagen']['labels_path']     = img_labels_paths[0]
C['val_datagen']['labels_path'] = img_labels_paths[1]
tg = init_session(C, make_classifier)
#%%
tg.train()
