# -*- coding: utf-8 -*-
import os
import random
import numpy as np

from pathlib import Path
from copy import deepcopy

from .util import Preprocessor, GenericPreprocessor, TimeseriesPreprocessor
from .util.data_loaders import DataLoader
from .util.configs import _DATAGEN_CFG
from .util.algorithms import ordered_shuffle
from .util._backend import WARN
from .util._default_configs import _DEFAULT_DATAGEN_CFG


class DataGenerator():
    """Central interface between a directory and `TrainGenerator`. Handles data
    loading, preprocessing, shuffling, and batching. Requires only
    `data_path` to run.

    Arguments:
        data_path: str
            Path to directory to load data from.
        batch_size: int
            Number of samples to feed the model at once. Can differ from
            size of batches of loaded files; see "Dynamic batching".
        labels_path: str / None
            Path to labels file. If None, will not load `labels`; can be used
            with `TrainGenerator.input_as_labels = True`, feeding `batch` as
            `labels` in :meth:`TrainGenerator.get_data` (e.g. autoencoders).
        preprocessor: None / custom object / str in ('timeseries',)
            Transforms `batch` and `labels` right before both are returned by
            :meth:`.get`. See :meth:`_set_preprocessor`.

            - str: fetches one of API-supported preprocessors.
            - None, uses :class:`GenericPreprocessor`.
            - Custom object: must subclass :class:`Preprocessor`.

        preprocessor_configs: None / dict
            Kwargs to pass to `preprocessor` in case it's None, str, or an
            uninstantiated custom object. Ignored if `preprocessor` is
            instantiated.
        data_loader: None / function / :class:`DataLoader`
            Object for loading data from directory / file.

                - function: passed as `loader` to `DataLoader.__init__` in
                  :meth:`_infer_and_set_info`; input signature: `(self, set_num)`
                - :class:`DataLoader` instance: will set `data_loader` directly
                - Class subclassing :class:`DataLoader` (uninstantiated):
                  will instantiate with attrs from :meth:`_infer_info` & others
                - str: name of one of loaders in :mod:`util.data_loaders`
                - None: defaults to one of defined in :mod:`util.data_loaders`,
                  as determined by :meth:`_infer_info`

        labels_loader: None / function / :class:`DataLoader`
            `data_loader`, but for labels.
        preload_labels: bool / None
            Whether to load all labels into `all_labels` at `__init__`. Defaults
            to True if `labels_path` is a file or a directory containing a
            single file.
        shuffle: bool
            If True, :meth:`reset_state` will shuffle `set_nums_to_process`;
            the method is called by `_on_epoch_end` within
            :meth:`TrainGenerator._train_postiter_processing` and
            :meth:`TrainGenerator._val_postiter_processing`
            (via :meth:`on_epoch_end`).
        superbatch_path: str / None
            Path to file or directory from which to load `superbatch`
            (:meth:`preload_superbatch`); see "Control Flow".
        set_nums: list[int] / None
            Used to set `set_nums_original` and `set_nums_to_process`. If None,
            will infer from `data_path`; see "Control Flow".
        superbatch_set_nums: list[int] / None
            set_nums to load into `superbatch`; see "Control Flow".

    **How it works**:

    Data is fed to :class:`TrainGenerator` via :class:`DataGenerator`. To work,
    data:

        - must be in one directory (or one file with all data)
        - file extensions must be same (.npy, .h5, etc)
        - file names must be enumerated with a common name
          (data1.npy, data2.npy, ...)
        - file batch size (# of samples, or dim 0 slices) should be same, but
          can also be in integer or fractal multiples of (x2, x3, x1/2, x1/3, ...)
        - labels must be in one file - unless feeding input as labels (e.g.
          autoencoder), which doesn't require labels files; just pass
          `TrainGenerator(input_as_labels=True)`

    **Flexible batch_size**:

    Loaded file's batch size may differ from `batch_size`, so long as former
    is an integer or integer fraction multiple of latter. Ex:

        - `len(loaded) == 32`, `batch_size == 64` -> will load another file
          and concatenate into `len(batch) == 64`.
        - `len(loaded) == 64`, `batch_size == 32` -> will set first half of
          `loaded` as `batch` and cache `loaded`, then repeat for second half.
        - 'Technically', files need not be integer (/ fraction) multiples, as
          the following load order works with `batch_size == 32`:
          `len(loaded) == 31`, `len(loaded) == 1` - but this is *not* recommended,
          as it's bound to fail if using shuffling, or if total number of
          samples isn't divisible by `batch_size`. Other problems may also arise.

    **Control Flow**:

        - `set_num`: index / Mapping key used to identify and get `batch` and
          `labels` via :meth:`load_data` and :meth:`load_labels`. Ex: for
          :meth:`DataLoader.numpy_loader`, which expects files shaped
          `(batches, samples, *)`, it'd do `np.load(path)[set_num]`.
        - `set_nums_to_process`: will pop `set_num` from this list until it's
          empty; once empty, will set `all_data_exhausted=True`
          (:meth:`update_state`).
        - `set_nums_original`: will reset `set_nums_to_process` to this with
          :meth:`reset_state`. It's either `set_nums` passed to `__init__`
          or is inferred in :meth:`_set_set_nums` as all available `set_nums`
          in `data_path` file / directory.
        - `superbatch`: dict of `set_num`-`batch`es loaded persistently in memory
          (RAM) as opposed to `batch`, which is overwritten. Once loaded, `batch`
          can be drawn straight from `superbatch` if `set_num` is in it (i.e.
          `superbatch_set_nums`).
        - :meth:`get` returns `batch` and `labels` fed through
          :meth:`Preprocessor.process`.
        - :meth:`advance_batch` gets "next" `batch` and `labels`. "Next" is
          determined by `set_num`, which is popped from `set_nums_to_process[0]`.
        - `batch_exhausted`: signals :class:`TrainGenerator` that a batch was
          consumed; this information is set via :meth:`update_state` per
          `_on_iter_end` within :meth:`TrainGenerator._train_postiter_processing`
          or :meth:`TrainGenerator._val_postiter_processing`.
        - If using slices (`slices_per_batch is not None`), then `batch_exhausted`
          is set to True only when `slice_idx == slices_per_batch - 1`.

    `__init__`:

    Instantiation. ("+" == if certain conditions are met)

        - +Infers missing configs based on args
        - Validates args & kwargs, and tries to correct, printing a"NOTE" or
          "WARNING" message where appropriate
        - +Preloads all labels into `all_labels`
        - Instantiates misc internal parameters to predefiend values (may be
          overridden by `TrainGenerator` loading).
    """
    _BUILTINS = {'preprocessors': (GenericPreprocessor, TimeseriesPreprocessor),
                 'loaders': DataLoader._BUILTINS,
                 'extensions': {'.npy', '.h5', '.csv'}}

    def __init__(self, data_path,
                 batch_size=32,
                 labels_path=None,
                 preprocessor=None,
                 preprocessor_configs=None,
                 data_loader=None,
                 labels_loader=None,
                 preload_labels=None,
                 shuffle=False,
                 superbatch_path=None,
                 set_nums=None,
                 superbatch_set_nums=None,
                 **kwargs):
        self.data_path=data_path
        self.batch_size=batch_size
        self.labels_path=labels_path
        self.preprocessor=preprocessor
        self.preprocessor_configs=preprocessor_configs or {}
        self.preload_labels=preload_labels
        self.shuffle=shuffle

        if superbatch_set_nums == 'all':
            if superbatch_path is not None:
                print(WARN, "will override `superbatch_path` with `data_path` "
                      "when `superbatch_set_nums == 'all'`")
            self.superbatch_path = data_path
        else:
            self.superbatch_path = superbatch_path

        self._init_and_validate_kwargs(kwargs)
        self._infer_and_set_info(data_loader, labels_loader)

        self._set_set_nums(set_nums, superbatch_set_nums)
        self._set_preprocessor(preprocessor, self.preprocessor_configs)

        if self.preload_labels:
            self.preload_all_labels()
        else:
            self.all_labels = {}
        self.labels = []  # initialize empty

        self._init_class_vars()
        print("DataGenerator initiated\n")

    ###### MAIN METHODS #######################################################
    def get(self, skip_validation=False):
        """Returns `(batch, labels)` fed to :meth:`Preprocessor.process`.

        skip_validation: bool
            - False (default): calls :meth:`_validate_batch`, which will
              :meth:`advance_batch` if `batch_exhausted`, and :meth:`reset_state`
              if `all_data_exhausted`.
            - True: fetch preprocessed `(batch, labels)` without advancing
              any internal states.
        """
        if not skip_validation:
            self._validate_batch()
        return self.preprocessor.process(self.batch, self.labels)

    def advance_batch(self, forced=False, is_recursive=False):
        """Sets next `batch` and `labels`; handles dynamic batching.

            - If `batch_loaded` and not `forced` (and not `is_recursive`),
              prints a warning that batch is loaded, and returns (does nothing)
            - `len(batch) != batch_size`:
                - `< batch_size`: calls :meth:`advance_batch` with
                  `is_recursive = True`. With each such call, `batch` and `labels`
                  are extended (stacked) until matching `batch_size`.
                - `> batch_size`, not integer multiple: raises `Exception`.
                - `> batch_size`, is integer multiple: makes `_group_batch` and
                  `_group_labels`, which are used to set `batch` and `labels`.

            - +If `set_nums_to_process` is empty, will raise `Exception`; it must
              have been reset beforehand via e.g. :meth:`reset_state`. If it's
              not empty, sets `set_num` by popping from `set_nums_to_process`.
              (+: only if `_group_batch` is None)
            - Sets or extends `batch` via :meth:`_get_next_batch` (by loading,
              or from `_group_batch` or `superbatch`).
            - +Sets or extends `labels` via :meth:`_get_next_labels` (by loading,
              or from `_group_labels`, or `all_labels`).
              (+: only if `labels_path` is a path (and not None))
            - Sets `set_name`, used by :class:`TrainGenerator` to print iteration
              messages.
            - Sets `batch_loaded = True`, `batch_exhausted = False`,
              `all_data_exhausted = False`, and `slice_idx` to None if it's
              already None (else to `0`).
        """
        def _handle_batch_size_mismatch(forced, is_recursive):
            if len(self.batch) < self.batch_size:
                self._update_set_name(is_recursive)
                self.advance_batch(forced, is_recursive=True)
                return 'exit'

            n_batches = len(self.batch) / self.batch_size
            if n_batches.is_integer():
                self._make_group_batch_and_labels(n_batches)
            else:
                raise Exception(f"len(batch) = {len(self.batch)} exceeds "
                                "`batch_size` and is its non-integer multiple")

        if not is_recursive:  # recursion is for stacking to match `batch_size`
            if self.batch_loaded and not forced:
                print(WARN, "'batch_loaded'==True; advance_batch() does "
                      "nothing\n(to force next batch, set 'forced'=True')")
                return
            self.batch = []
            self.labels = []

        if self._group_batch is None:
            if len(self.set_nums_to_process) == 0:
                raise Exception("insufficient samples (`set_nums_to_process` "
                                "is empty)")
            self.set_num = self.set_nums_to_process.pop(0)
            self._set_names = [str(self.set_num)]

        self.batch.extend(self._get_next_batch())
        self.labels.extend(self._get_next_labels())
        if self._group_batch is not None:
            self._update_group_batch_state()

        if self.batch_size is not None and len(self.batch) != self.batch_size:
            flag = _handle_batch_size_mismatch(forced, is_recursive)
            if flag == 'exit':
                # exit after completing arbitrary number of recursions, by
                # which time code below would execute as needed
                return

        self._update_set_name(is_recursive)
        self.batch = np.asarray(self.batch)
        if len(self.labels) > 0:
            self.labels = np.asarray(self.labels)

        self.batch_loaded = True
        self.batch_exhausted = False
        self.all_data_exhausted = False
        self.slice_idx = None if self.slice_idx is None else 0

    ###### MAIN METHOD HELPERS ################################################
    def _get_next_batch(self, set_num=None, warn=True):
        """Gets `batch` per `set_num`.

            - `set_num = None`: will use `self.set_num`.
            - `warn = False`: won't print warning on superbatch not being
              preloaded.
            - If `_group_batch` is not None, will get `batch` from `_group_batch`.
            - If `set_num` is in `superbatch_set_nums`, will get `batch`
              as `superbatch[set_num]` (if `superbatch` exists).
            - By default, gets `batch` via :meth:`load_data`.
        """
        set_num = set_num or self.set_num

        if self._group_batch is not None:
            batch = self._batch_from_group_batch()
        elif set_num in self.superbatch_set_nums:
            if self.superbatch is not None and len(self.superbatch) > 0:
                if set_num in self.superbatch:
                    batch = self.superbatch[set_num]
                else:
                    if warn:
                        print(WARN, f"`set_num` ({set_num}) found in "
                              "`superbatch_set_nums` but not in `superbatch`; "
                              "will use `load_data` instead.")
                    batch = self.load_data(set_num)
            else:
                if warn:
                    print(WARN, f"`set_num` ({set_num}) found in `superbatch_"
                          "set_nums` but `superbatch` is empty; call "
                          "`preload_superbatch()`")
                batch = self.load_data(set_num)
        else:
            batch = self.load_data(set_num)
        return batch

    def _get_next_labels(self, set_num=None):
        """Gets `labels` per `set_num`.

            - `set_num = None`: will use `self.set_num`.
            - If `_group_labels` is not None, will get `labels` from
              `_group_labels`.
            - If `set_num` is in `superbatch_set_nums`, will get `batch`
              as `superbatch[set_num]` (if `superbatch` exists).
            - By default, gets `labels` via :meth:`load_data`, if `labels_path`
              is set - else, `labels=[]`.
        """
        set_num = set_num or self.set_num

        if not self.labels_path:
            labels = []
        elif self._group_labels is not None:
            labels = self._labels_from_group_labels()
        elif set_num in self.all_labels:
            labels = self.all_labels[set_num]
        else:
            labels = self.load_labels(set_num)
        return labels

    def _batch_from_group_batch(self):
        """Slices `_group_batch` per `batch_size` and `_group_batch_idx`."""
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_batch[start:end]

    def _labels_from_group_labels(self):
        """Slices `_group_labels` per `batch_size` and `_group_batch_idx`."""
        start = self.batch_size * self._group_batch_idx
        end = start + self.batch_size
        return self._group_labels[start:end]

    def _update_group_batch_state(self):
        """Sets "group" attributes to `None` once sufficient number of batches
        were extracted, else increments `_group_batch_idx`.
        """
        if (self._group_batch_idx + 1
            ) * self.batch_size == len(self._group_batch):
            self._group_batch = None
            self._group_labels = None
            self._group_batch_idx = None
        else:
            self._group_batch_idx += 1

    def _update_set_name(self, is_recursive):
        s = self._set_names.pop(0)
        self.set_name = s if not is_recursive else "%s+%s" % (self.set_name, s)

    ###### STATE METHOS #######################################################
    def on_epoch_end(self):
        """Increments `epoch`, calls `preprocessor.on_epoch_end(epoch)`, then
        :meth:`reset_state`, and returns `epoch`.
        """
        self.epoch += 1
        self.preprocessor.on_epoch_end(self.epoch)
        self.reset_state()
        return self.epoch

    def update_state(self):
        """Calls `preprocessor.update_state()`, and if `batch_exhausted` and
        `set_nums_to_process == []`, sets `all_data_exhausted = True` to signal
        :class:`TrainGenerator` of epoch end.
        """
        self.preprocessor.update_state()
        if self.batch_exhausted and self.set_nums_to_process == []:
            self.all_data_exhausted = True

    def reset_state(self, shuffle=None):
        """Calls `preprocessor.reset_state()`, sets `batch_exhausted = True`,
        `batch_loaded = False`, resets `set_nums_to_process` to
        `set_nums_original`, and shuffles `set_nums_to_process` if `shuffle`.

            - If `shuffle` passed in is None, will set from `self.shuffle`.
            - Used in :meth:`TrainGenerator.reset_validation` w/ `shuffle=False`.
        """
        self.preprocessor.reset_state()
        # ensure below values prevail, in case `preprocessor` sets them to
        # something else; also sets `preprocessor` attributes
        self.batch_exhausted = True
        self.batch_loaded = False
        self.set_nums_to_process = self.set_nums_original.copy()

        shuffle = shuffle or self.shuffle
        if shuffle:
            random.shuffle(self.set_nums_to_process)
            print('\nData set_nums shuffled\n')

    ###### MISC METHOS ########################################################
    def _validate_batch(self):
        """If `all_data_exhausted`, calls :meth:`reset_state`.
        If `batch_exhausted`, calls :meth:`advance_batch`.
        """
        if self.all_data_exhausted:
            print(WARN, "all data exhausted; automatically resetting "
                  "datagen state")
            self.reset_state()
        if self.batch_exhausted:
            print(WARN, "batch exhausted; automatically advancing batch")
            self.advance_batch()

    def _make_group_batch_and_labels(self, n_batches):
        """Makes `_group_batch` and `_group_labels` when loaded `len(batch)`
        exceeds `batch_size` as its integer multiple. May shuffle.

            - `_group_batch = np.asarray(batch)`, and
              `_group_labels = np.asarray(labels)`; each's `len() > batch_size`.
            - Shuffles if:
                - `shuffle_group_samples`: shuffles all samples (dim0 slices)
                - `shuffle_group_batches`: groups dim0 slices by `batch_size`,
                  then shuffles the groupings. Ex:

                  >>> batch_size == 32
                  >>> batch.shape == (128, 100)
                  >>> batch = batch.reshape()  # (4, 32, 100) == .shape
                  >>> shuffle(batch)           # 24 (4!) permutations
                  >>> batch = batch.reshape()  # (128, 100)   == .shape

            - Sets `_group_batch_idx = 0`, and calls
              :meth:`_update_group_batch_state`.
            - Doesn't affect `labels` if `labels_path` is falsy (e.g. None)
        """
        def _maybe_shuffle(gb, lb=None):
            if lb is not None:
                if self.shuffle_group_samples:
                    shuffled = ordered_shuffle(gb, lb)
                elif self.shuffle_group_batches:
                    gb_shape, lb_shape = gb.shape, lb.shape
                    gb = gb.reshape(-1, self.batch_size, *gb_shape[1:])
                    lb = lb.reshape(-1, self.batch_size, *lb_shape[1:])
                    gb, lb = ordered_shuffle(gb, lb)
                    shuffled = gb.reshape(*gb_shape), lb.reshape(*lb_shape)
                else:
                    shuffled = gb, lb  # no shuffle
            else:
                if self.shuffle_group_samples:
                    np.random.shuffle(gb)
                    shuffled = gb
                elif self.shuffle_group_batches:
                    gb_shape = gb.shape
                    gb = gb.reshape(-1, self.batch_size, *gb_shape[1:])
                    np.random.shuffle(gb)
                    shuffled = gb.reshape(*gb_shape)
                else:
                    shuffled = gb  # no shuffle
            return shuffled

        self._set_names = [f"{self.set_num}-{postfix}" for postfix in
                           "abcdefghijklmnopqrstuvwxyz"[:int(n_batches)]]

        gb = np.asarray(self.batch)
        if self.labels_path:
            lb = np.asarray(self.labels)
            if len(gb) != len(lb):
                raise Exception(("len(batch) != len(labels) ({} != {})"
                                ).format(len(gb), len(lb)))

        self.batch = []  # free memory
        if self.labels_path:
            self.labels = []  # free memory

            gb, lb = _maybe_shuffle(gb, lb)
        else:
            gb = _maybe_shuffle(gb)

        self._group_batch = gb
        if self.labels_path:
            self._group_labels = lb
        self._group_batch_idx = 0

        self.batch = self._batch_from_group_batch()
        if self.labels_path:
            self.labels = self._labels_from_group_labels()
        self._update_group_batch_state()

    ###### PROPERTIES #########################################################
    @property
    def batch_exhausted(self):
        """Is retrieved from and set in `preprocessor`.
        Indicates that `batch` and `labels` for given `set_num` were consumed
        by `TrainGenerator` (if using slices, that all slices were consumed).

        Ex: `self.batch_exhausted = 5` will set
        `self.preprocessor.batch_exhausted = 5`, and `print(self.batch_exhausted)`
        will then print `5` (or something else if `preprocessor` changes it
        internally).
        """
        return self.preprocessor.batch_exhausted

    @batch_exhausted.setter
    def batch_exhausted(self, value):
        self.preprocessor.batch_exhausted = value

    @property
    def batch_loaded(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        Indicates that `batch` and `labels` for given `set_num` are loaded.
        """
        return self.preprocessor.batch_loaded

    @batch_loaded.setter
    def batch_loaded(self, value):
        self.preprocessor.batch_loaded = value

    @property
    def slices_per_batch(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        """
        return self.preprocessor.slices_per_batch

    @slices_per_batch.setter
    def slices_per_batch(self, value):
        self.preprocessor.slices_per_batch = value

    @property
    def slice_idx(self):
        """Is retrieved from and set in `preprocessor`, same as `batch_exhausted`.
        """
        return self.preprocessor.slice_idx

    @slice_idx.setter
    def slice_idx(self, value):
        self.preprocessor.slice_idx = value

    @property
    def load_data(self):
        """Load and return `batch` data via
        :meth:`data_loaders.DataLoader.load_fn`.
        Used by :meth:`_get_next_batch` and :meth:`preload_superbatch`.
        """
        return self.data_loader.load_fn

    @load_data.setter
    def load_data(self, loader):
        self.data_loader.load_fn = loader

    @property
    def load_labels(self):
        """Load and return `labels` data via
        :meth:`data_loaders.DataLoader.load_fn`.
        Used by :meth:`_get_next_labels` and :meth:`preload_all_labels`.
        """
        return self.labels_loader.load_fn

    @load_labels.setter
    def load_labels(self, loader):
        self.labels_loader.load_fn = loader

    ###### INIT METHODS #######################################################
    def _infer_info(self, path):
        """Infers unspecified essential attributes from directory and contained
        files info:

            - Checks that the data directory (`path`) isn't empty
              (files whose names start with `'.'` aren't counted)
            - Retrieves data filepaths per `path` and gets data extension (to
              most frequent ext in dir, excl. "other path" from count if in same
              dir. "other path" is `data_path` if `path == labels_path`, and
              vice versa.)
            - Gets `base_name` as longest common substring among files with
              `ext` extension
            - If `path` is path to a file, then `filepaths=[path]`.
            - If `path` is None, returns `base_name=None` `ext=None`,
              `filepaths=[]`.
        """
        def _validate_directory(path):
            # not guaranteed to catch hidden files
            nonhidden_files_names = [x for x in os.listdir(path)
                                     if not x.startswith('.')]
            if len(nonhidden_files_names) == 0:
                raise Exception("`path` is empty (%s)" % path)

        def _get_base_name(filepaths):
            def _longest_common_substr(data):
                # longest substring common to all filenames
                substr = ''
                ref = data[0]
                for i in range(len(ref)):
                  for j in range(len(ref) - i + 1):
                    if j > len(substr) and all(ref[i:i + j] in x for x in data):
                      substr = ref[i:i + j]
                return substr

            filenames = [Path(p).stem for p in filepaths]
            base_name = _longest_common_substr(filenames)
            return base_name

        def _get_filepaths_and_ext(path):
            def _infer_extension(path, other_path):
                supported = DataGenerator._BUILTINS['extensions']
                extensions = [p.suffix for p in Path(path).iterdir()
                              if (p.suffix in supported and str(p) != other_path)]
                if len(extensions) == 0:
                    raise Exception("No files found with supported extensions: "
                                    + ', '.join(supported) + " in `path` ", path)
                # pick most frequent extension
                ext = max(set(extensions), key=extensions.count)

                if len(set(extensions)) > 1:
                    print(WARN, "multiple file extensions found in "
                          "`path`; only", ext, "will be used ")
                return ext

            other_path = (self.labels_path if path == self.data_path else
                          self.data_path)
            ext = _infer_extension(path, other_path)
            filepaths = [str(p) for p in Path(path).iterdir()
                         if (p.suffix == ext and str(p) != other_path)]
            return filepaths, ext

        if path is None:
            return dict(base_name=None, ext=None, filepaths=[])
        elif os.path.isdir(path):
            _validate_directory(path)
            filepaths, ext = _get_filepaths_and_ext(path)
            print("Discovered %s files with matching format" % len(filepaths))
        else:  # isfile
            print("Discovered dataset with matching format")
            filepaths = [path]
            ext = Path(path).suffix

        base_name = _get_base_name(filepaths)
        return dict(base_name=base_name, ext=ext, filepaths=filepaths)

    def _infer_and_set_info(self, data_loader, labels_loader):
        """Sets `data_loader` and `labels_loader` (:class:`DataLoader`), using
        info obtained from :meth:`_infer_info`.

            - If `info` contains only one filepath, loader will operate with
              `_is_dataset=True`.
            - If `preload_labels` is None and `labels_loader._is_dataset`,
              will set `preload_labels=True`.

        `data_loader` / `labels_loader` are:

            - function: passed as `loader` to `DataLoader.__init__`;
              input signature: `(self, set_num)`
            - :class:`DataLoader` instance: will set `data_loader` directly
            - Class subclassing :class:`DataLoader` (uninstantiated):
              will instantiate with attrs from :meth:`_infer_info` & others
            - str: name of one of loaders in :mod:`util.data_loaders`
            - None: defaults to one of defined in :mod:`util.data_loaders`,
              as determined by :meth:`_infer_info`
        """
        def _set_loaders(data_loader, labels_loader):
            # set explicitly to index by linter
            if isinstance(data_loader, DataLoader):
                self.data_loader = data_loader
            else:
                kw = dict(path=self.data_path,
                          dtype=self.data_dtype,
                          batch_shape=self.data_batch_shape,
                          **self._infer_info(self.data_path))
                if isinstance(data_loader, type):
                    if not issubclass(data_loader, DataLoader):
                        raise TypeError("`data_loader` class must subclass "
                                        "`DataLoader` (got %s)" % data_loader)
                    self.data_loader = data_loader(loader=None, **kw)
                else:  # function / str / None
                    self.data_loader = DataLoader(loader=data_loader, **kw)

            if isinstance(labels_loader, DataLoader):
                self.labels_loader = labels_loader
            else:
                kw = dict(path=self.labels_path,
                          dtype=self.labels_dtype,
                          batch_shape=self.labels_batch_shape,
                          **self._infer_info(self.labels_path))
                if isinstance(labels_loader, type):
                    if not issubclass(labels_loader, DataLoader):
                        raise TypeError("`labels_loader` class must subclass "
                                        "`DataLoader` (got %s)" % labels_loader)
                    self.labels_loader = labels_loader(loader=None, **kw)
                else:  # function / None
                    self.labels_loader = DataLoader(loader=labels_loader, **kw)

        _set_loaders(data_loader, labels_loader)

        if self.preload_labels is None and self.labels_loader._is_dataset:
            self.preload_labels = True

    def _set_set_nums(self, set_nums, superbatch_set_nums):
        """Sets `set_nums_original`, `set_nums_to_process`, and
        `superbatch_set_nums`.

            - Fetches `set_nums` via :meth:`DataLoader._get_set_nums`
            - Sets `set_nums_to_process` and `set_nums_original`; if `set_nums`
              weren't passed to `__init__`, sets to fetched ones.
            - If `set_nums` were passed, validates that they're a subset of
              fetched ones (i.e. can be seen by `data_loader`).
            - Sets `superbatch_set_nums`; if not passed to `__init__`,
              and `== 'all'`, sets to fetched ones. If passed, validates that
              they subset fetched ones.
            - Does not validate set_nums from `labels_loader`'s perspective;
              user is expected to supply a `labels` to each `batch` with common
              `set_num`.
        """
        def _set_and_validate_set_nums(set_nums):
            nums_to_process = self.data_loader._get_set_nums()

            if not set_nums:
                self.set_nums_original   = nums_to_process.copy()
                self.set_nums_to_process = nums_to_process.copy()
                print(len(nums_to_process), "set nums inferred; if more are "
                      "expected, ensure file names contain a common substring "
                      "w/ a number (e.g. 'train1.npy', 'train2.npy', etc)")
            else:
                if any((num not in nums_to_process) for num in set_nums):
                    raise Exception("a set_num in `set_nums_to_process` was not "
                                    "in set_nums found from "
                                    "`data_path` filenames")
                self.set_nums_original   = set_nums.copy()
                self.set_nums_to_process = set_nums.copy()

        def _set_and_validate_superbatch_set_nums(superbatch_set_nums):
            if superbatch_set_nums != 'all' and not self.superbatch_path:
                if superbatch_set_nums:
                    print(WARN, "`superbatch_set_nums` will be ignored, "
                          "since `superbatch_path` is None")
                self.superbatch_set_nums = []
                return

            nums_to_process = self.data_loader._get_set_nums()

            if (superbatch_set_nums == 'all' or
                (superbatch_set_nums is None and self.superbatch_path)):
                self.superbatch_set_nums = nums_to_process.copy()
            else:
                if any(num not in nums_to_process for num in superbatch_set_nums):
                    raise Exception("a `set_num` in `superbatch_set_nums` "
                                    "was not in set_nums found from "
                                    "`superbatch_folderpath` filename")
                self.superbatch_set_nums = superbatch_set_nums

        _set_and_validate_set_nums(set_nums)
        _set_and_validate_superbatch_set_nums(superbatch_set_nums)

    def _set_preprocessor(self, preprocessor, preprocessor_configs):
        """Sets `preprocessor`, based on `preprocessor` passed to `__init__`:

            - If None, sets to :class:`GenericPreprocessor`, instantiated with
              `preprocessor_configs`.
            - If an uninstantiated class, will validate that it subclasses
              :class:`Preprocessor`, then isntantiate with `preprocessor_configs`.
            - If string, will match to a supported builtin.
            - Validates that the set `preprocessor` subclasses
              :class:`Preprocessor`.
        """
        def _set(preprocessor, preprocessor_configs):
            if preprocessor is None:
                self.preprocessor = GenericPreprocessor(**preprocessor_configs)
            elif isinstance(preprocessor, type):  # uninstantiated
                self.preprocessor = preprocessor(**preprocessor_configs)
            elif preprocessor == 'timeseries':
                self.preprocessor = TimeseriesPreprocessor(**preprocessor_configs)
            else:
                self.preprocessor = preprocessor
            if not isinstance(self.preprocessor, Preprocessor):
                raise TypeError("`preprocessor` must subclass `Preprocessor`")

        _set(preprocessor, preprocessor_configs)
        self.preprocessor._validate_configs()

    def preload_superbatch(self):
        """Loads all data specified by `superbatch_set_nums` via
        :meth:`load_data`, and assigns them to `superbatch` for each `set_num`.
        """
        def _set_superbatch_attrs():
            # get and set `superbatch` variant of data attributes:
            # loader, base_name, ext, filenames, filepaths
            info = self._infer_info(self.superbatch_path)
            for name in info:
                alias = '_superbatch_' + name
                if name in ('filenames', 'filepaths'):
                    # include filenames & filepaths only if they're actually used
                    setattr(self, alias, [])
                    for f in info[name]:
                        set_num = Path(f).stem.split(info['base_name'])[-1]
                        if set_num in self.superbatch_set_nums:
                            getattr(self, alias).append(f)
                else:
                    setattr(self, alias, info[name])

        print(end='Preloading superbatch ... ')
        _set_superbatch_attrs()

        self.superbatch = {}  # empty if not empty
        for set_num in self.superbatch_set_nums:
            self.superbatch[set_num] = self.load_data(set_num)
            print(end='.')

        num_samples = sum(len(batch) for batch in self.superbatch.values())
        print(" finished, w/", num_samples, "total samples")

    def preload_all_labels(self):
        """Loads all labels into `all_labels` using :meth:`load_labels`, based
        on `set_nums_original`.
        """
        self.all_labels = {}  # empty if not empty
        for set_num in self.set_nums_original:
            self.all_labels[set_num] = self.load_labels(set_num)

    def _init_and_validate_kwargs(self, kwargs):
        """Sets and validates `kwargs` passed to `__init__`.

            - Ensures `data_path` is a file or a directory, and `labels_path`
              is a file, directory, or None.
            - Ensures kwargs are functional (compares against names in
              :data:`~deeptrain.util._default_configs._DEFAULT_DATAGEN_CFG`.
            - Sets whichever names were passed with `kwargs`, and defaults
              the rest.
        """
        def _validate_data_and_labels_path():
            def is_file_or_dir(x):
                return isinstance(x, str) and (os.path.isfile(x) or
                                               os.path.isdir(x))
            if not is_file_or_dir(self.data_path):
                raise ValueError("`data_path` must be a file or a directory "
                                 f"(got {self.data_path})")
            if not (is_file_or_dir(self.labels_path) or self.labels_path is None):
                raise ValueError("`labels_path` must be a file, a directory, or "
                                 f"None (got {self.labels_path})")

        def _validate_kwarg_names(kwargs):
            for kw in kwargs:
                if kw not in _DEFAULT_DATAGEN_CFG:
                    raise ValueError("unknown kwarg: '{}'".format(kw))

        def _set_kwargs(kwargs):
            class_kwargs = deepcopy(_DATAGEN_CFG)
            class_kwargs.update(kwargs)

            for attribute in class_kwargs:
                setattr(self, attribute, class_kwargs[attribute])

        def _validate_shuffle_group_():
            if self.shuffle_group_batches and self.shuffle_group_samples:
                print(WARN, "`shuffle_group_batches` will be ignored since "
                      "`shuffle_group_samples` is also ==True")

        _validate_data_and_labels_path()
        _validate_kwarg_names(kwargs)
        _set_kwargs(kwargs)
        _validate_shuffle_group_()

    def _init_class_vars(self):
        """Instantiates various internal attributes. Most of these are saved
        and loaded by :class:`TrainGenerator` by default."""
        _defaults = dict(
            all_data_exhausted=False,
            batch_exhausted=True,
            batch_loaded=False,
            epoch=0,  # managed externally
            superbatch={},
            _group_batch=None,
            _group_labels=None,
            set_num=None,
            set_name=None,
            _set_names=[],
            start_increment=0,
            )
        for k, v in _defaults.items():
            setattr(self, k, getattr(self, k, v))

        # used in saving & report generation
        self._path_attrs = ['data_path', 'labels_path', 'superbatch_path']
