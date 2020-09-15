# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - Creating custom callbacks
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()
from utils import make_classifier, init_session, img_labels_paths
from utils import CL_CONFIGS as C
from deeptrain.callbacks import TraingenCallback

import matplotlib.pyplot as plt

#%%#
# We can use two types of callbacks: objects (instances of TraingenCallback),
# or functions.

# Function callback takes TrainGenerator instance as the only argument.
# Below we print the total number of batches fit so far
def print_batches_fit(tg):
    print("\nBATCHES FIT: %s\n" % tg._batches_fit)

# The next step is to specify *when* the callback is called. Callbacks are
# called at several stages throughout training:
#   {'train:iter', 'train:batch', 'train:epoch',
#    'val:iter',   'val:batch',   'val:epoch',
#    'val_end', 'save', 'load'}
# 'train:batch', for example, corresponds to `_on_batch_end` within
# `_train_postiter_processing` (TrainGenerator methods).
pbf = {'train:epoch': print_batches_fit}  # print on every epoch
#%%#
# Callback objects subclass TraingenCallback, which defines methods to
# override as ways to specify the *when* instead of dict keys.
# See deeptrain.callbacks.TraingenCallback.

# Show histogram of first layer's kernel weights at end of each validation
class VizWeights(TraingenCallback):
    def on_val_end(self, stage=None):
        # method will be called within TrainGenerator._on_val_end
        W = self.tg.model.layers[1].get_weights()[0]
        plt.hist(W.ravel(), bins=200)
        plt.show()

vizw = VizWeights()
#%%
C['traingen']['epochs'] = 4
C['traingen']['callbacks'] = [pbf, vizw]
C['datagen']['labels_path']     = img_labels_paths[0]
C['val_datagen']['labels_path'] = img_labels_paths[1]
tg = init_session(C, make_classifier)
#%%
tg.train()
