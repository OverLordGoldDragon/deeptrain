# -*- coding: utf-8 -*-
"""This example assumes you've read `callbacks/basic.py`, and covers:
    - Setting and restoring random seeds at arbitrary frequency for restoring
    from (nearly) any point in training
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import make_classifier, init_session
from utils import CL_CONFIGS as C
from deeptrain.callbacks import RandomSeedSetter

#%%#
# Set new random seeds (`random`, `numpy`, TF-graph, TF-global) every epoch,
# incrementing by 1 from start value (default 0).
# Since `tg.save()` is called each epoch, we specify `freq` via `'save'`
# instead of `'train:epoch'`.
seed_freq = {'save': 1, 'load': 1}
seed_setter = RandomSeedSetter(freq=seed_freq)
#%%###########################################################################
C['traingen']['callbacks'] = [seed_setter]
C['traingen']['epochs'] = 3
C['traingen']['iter_verbosity'] = 0
tg = init_session(C, make_classifier)
#%%
tg.train()
#%%
# Text printed after epoch shows the values each of the four random seed
# were set to, which by default start at 0 and increment by 1.
# Note that often it's printed and incremented twice, since `tg.save()` is
# also called within `tg._save_best_model()`.
# Note further that TensorFlow lacks a global random state for later recovery
# (though it's possible to achieve with meticulous model & graph definition).
# Setting the seed at a point, and then loading the point and setting it again
# (which is what we'll do), however, works.
#%%# Clear current session
# Retrieve last saved logfile to then load
loadpath = tg.get_last_log('state')
tg.destroy(confirm=True)
del tg, seed_setter  # seed_setter has internal reference to `tg`; destroy it
#%%# Start new session, load savefile
C['traingen']['loadpath'] = loadpath
C['traingen']['callbacks'] = [RandomSeedSetter(freq=seed_freq)]
tg = init_session(C, make_classifier)
# Last random seed loaded and set; same would apply if we loaded from
# an earlier epoch.
