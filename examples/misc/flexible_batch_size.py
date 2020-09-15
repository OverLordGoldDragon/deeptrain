# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - How batch_size can be a multiple of batch_size on file
    - Faster SSD loading via flexible batch size
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import make_autoencoder, init_session, AE_CONFIGS as C

#%% DeepTrain can use batch_size an integral multiple of one on file, by
# splitting up into smaller batches or combining into larger. If a file stores
# 128 samples, we can split it to x2 64-sample batches, or combine two
# files into x1 256-sample batch.
C['traingen']['epochs'] = 1
#%% User batch_size=64, file batch_size=128
C['datagen'    ]['batch_size'] = 64
C['val_datagen']['batch_size'] = 64
C['model']['batch_shape'] = (64, 28, 28, 1)
#%%
tg = init_session(C, make_autoencoder)
#%%
tg.train()
#%% User batch_size=256, file batch_size=128
C['datagen'    ]['batch_size'] = 256
C['val_datagen']['batch_size'] = 256
C['model']['batch_shape'] = (256, 28, 28, 1)
#%%
tg = init_session(C, make_autoencoder)
#%%
tg.train()
#%% We can see the difference in the two settings through sets logging:
#  - `batch_size=64`: a `set_num` is split into `'a'` and `'b'`
#  - `batch_size=256`: `set_num1 + set_num2`, combining two files
#%% Faster SSD Loading:
# - Save larger `batch_size` on disk (e.g. 512) than is used (e.g. 32).
# - Larger files much better utilize an SSD's read speed via parallelism.
# - `batch_size` on file can be as large as RAM permits.
