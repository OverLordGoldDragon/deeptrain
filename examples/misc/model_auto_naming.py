# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - How to configure automatic model naming
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import make_autoencoder, init_session, AE_CONFIGS as C

#%% DeepTrain auto-names model based on `model_name_configs`, a dict.
#  - Keys denote either TrainGenerator attributes, its object's attributes
#    (via `.`), or `model_configs` keys. `'best_key_metric'` reflects the actual
#    value, if `TrainGenerator` checkpointed since last change.
#  - Values denote attribute aliases; if blank or None, will use attrs as given.
name_cfg = {'datagen.batch_size': 'BS',
            'filters': 'filt',
            'optimizer': '',
            'lr': '',
            'best_key_metric': '__max'}
C['traingen'].update({'epochs': 1,
                      'model_base_name': "AE",
                      'model_name_configs': name_cfg})
C['model']['optimizer'] = 'Adam'
C['model']['lr'] = 1e-4
#%%
tg = init_session(C, make_autoencoder)
#%%
tg.train()
#%%
print(tg.model_name)
#%% Note that `logdir` and best model saves are also named with `model_name`;
# it, together with `model_num`, enables scalable reference to hundreds of
# trained models: sort through models by reading off key hyperparameters.
print(tg.logdir)
print(tg.get_last_log('best'))