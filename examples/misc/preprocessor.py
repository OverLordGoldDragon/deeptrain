# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
    - Creating custom Preprocessor
    - How train & val loop and DataGenerator logic can be changed
      via Preprocessor
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()

from utils import make_autoencoder, init_session, AE_CONFIGS as C
from deeptrain.util.preprocessors import Preprocessor
import numpy as np

#%%#
# Preprocessor communicates with DataGenerator twofold:
#  - .process() is called in DataGenerator.get()
#  - DataGenerator sets and gets following attributes *through* Preprocessor:
#    - `batch_exhausted`, `batch_loaded`, `slices_per_batch`, `slice_idx`
#  - Thus, Preprocessor can dictate train & validation loop logic by specifying
#    when a batch ends (setting `batch_exhausted`) in `.process()`, when
#    some condition holds
#%%
# Below preprocessor randomly crops images to a predefined width & height,
# as an example of `.process()` in action.
class RandCropPreprocessor(Preprocessor):
    """2D random crop. MNIST is 28x28, we try 25x25 crops,
    e.g. batch[2:27, 3:28]."""
    def __init__(self, size, crop_batch=True, crop_labels=False,
                 crop_same=False):
        # length          -> (length, length)
        # (width, height) -> (width, height)
        assert isinstance(size, (tuple, int))
        self.size = size if isinstance(size, tuple) else (size, size)

        self.crop_batch = crop_batch
        self.crop_labels = crop_labels
        self.crop_same = crop_same

    def process(self, batch, labels):
        if self.crop_batch:
            (x_start, x_end), (y_start, y_end) = self._make_crop_mask(batch)
            batch = batch[:, x_start:x_end, y_start:y_end]
        if self.crop_labels:
            if not self.crop_same or not self.crop_batch:
                (x_start, x_end), (y_start, y_end
                                   ) = self._make_crop_mask(labels)
            labels = labels[:, x_start:x_end, y_start:y_end]
        return batch, labels

    def _make_crop_mask(self, data):
        _, w, h, *_ = data.shape  # (samples, width, height, channels)
        x_offset = np.random.randint(0, w - self.size[0])
        y_offset = np.random.randint(0, h - self.size[1])
        x_start, x_end = x_offset, x_offset + self.size[0]
        y_start, y_end = y_offset, y_offset + self.size[1]
        return (x_start, x_end), (y_start, y_end)
#%%
C['datagen'    ]['preprocessor'] = RandCropPreprocessor(size=24)
C['val_datagen']['preprocessor'] = RandCropPreprocessor(size=24)
C['datagen'    ]['batch_size'] = 128
C['val_datagen']['batch_size'] = 128
C['model']['batch_shape'] = (128, 24, 24, 1)
C['traingen']['iter_verbosity'] = 0
C['traingen']['epochs'] = 1
#%%
tg = init_session(C, make_autoencoder)
#%%
tg.train()
#%%
# A better example of Preprocessor communicating with DataGenerator is the builtin
# `deeptrain.util.preprocessors.TimeseriesPreprocessor`, demonstrated in
# `examples/misc/timeseries`. Its main logic methods are worth inspecting.
#%%
#`.process()` checks if we're at the first slice (window), and sets
# the window sequence length and number of windows per batch accordingly.
# This enables having variable windows per batch.
from deeptrain.util.preprocessors import TimeseriesPreprocessor
from inspect import getsource
print(getsource(TimeseriesPreprocessor.process))
#%%# `._next_window()` fetches next window in the sequence according to
# `slice_idx`, `window_size`, and two other attrs (see docs)
print(getsource(TimeseriesPreprocessor._next_window))
#%%# Lastly, it tells DataGenerator that batch ends when the last window
# was processed:
print(getsource(TimeseriesPreprocessor.update_state))
# called within DataGenerator.update_state
