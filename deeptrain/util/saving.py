# -*- coding: utf-8 -*-
import os
import pickle
import types
import tensorflow as tf

from pathlib import Path
from ._backend import K, WARN
from ..visuals import get_history_fig
from .misc import pass_on_error, _init_optimizer
from .experimental import exclude_unpickleable
from ..backend import tensor_utils


def _save_best_model(self, del_previous_best=None):
    """Saves `model`, history fig, `TrainGenerator` state, and report
    if latest key metric is a new best (max/min, as per `max_is_best`).
    Also deletes previous best saves if `max_one_best_save` (as called by
    :meth:`TrainGenerator._on_val_end`, and defaulting to if None passed).
    """
    def _validate_is_best():
        """Enforce best model condition in case the method is called manually"""
        if self.max_is_best:
            new_best = (self.key_metric_history[-1] > self.best_key_metric)
        else:
            new_best = (self.key_metric_history[-1] < self.best_key_metric)
        if not new_best:
            raise Exception("`_save_best_model` was called with latest key_metric "
                            "not being a new best (new: {}, prev-best: {})"
                            ).format(self.best_key_metric,
                                     self.key_metric_history[-1])

    def _del_previous_best():
        def _get_prev_files():
            return [str(p) for p in Path(self.best_models_dir).iterdir()
                    if p.name.split('__')[0][1:] == str(self.model_num)]

        prev_files = _get_prev_files()
        for f in prev_files:
            os.remove(f)

    if del_previous_best is None:
        del_previous_best = bool(self.max_one_best_save)
    if del_previous_best:
        try:
            _del_previous_best()
        except BaseException as e:
            print(WARN,  "previous best model files could not be deleted; "
                  "skipping")
            print("Errmsg:", e)

    _validate_is_best()
    self.best_key_metric = round(self.key_metric_history[-1], 6)
    _update_best_key_metric_in_model_name(self)

    basepath = os.path.join(self.best_models_dir, self.model_name)
    save_fns = self._make_model_save_fns(basepath + '__')
    for path, save_fn in save_fns:
        save_fn(path)

    self._history_fig = get_history_fig(self)
    self._history_fig.savefig(basepath + '.png')

    self.save(basepath + '__state.h5')

    if self._imports.get('PIL', False):
        try:
            self.save_report(basepath + '__report.png')
        except BaseException as e:
            print(WARN, "Best model report could not be saved; skipping")
            print("Errmsg", e)
    print("Best model saved to " + basepath)

def checkpoint(self, forced=False, overwrite=None):
    """Saves `TrainGenerator` state (including both `DataGenerator`s), report,
    history fig, and model (weights and if configured to, optimizer state and
    architecture).

    Arguments:
        forced: bool
            If True, will checkpoint whether or not temp / unique checkpoint
            freq's were met.
        overwrite: bool / None
            If None, set from `checkpoints_overwrite_duplicates`. If True,
            will overwrite existing checkpoint files if having same name
            as one generated with current checkpoint - else, will make unique
            names by incrementing '_v2', '_v3', etc.

    Saves according to `temp_checkpoint_freq` and `unique_checkpoint_freq`.
    See :meth:`._should_do`. See :meth:`save` on how model and
    `TrainGenerator` state are save. Additionally, ill update `model_name`
    if `best_key_metric` was updated.

    "Unique" checkpoint will generate save files with latest `best_key_metric`
    and `_times_validated`, and with full `model_name` if
    `logs_use_full_model_name`.
    """
    def _get_savename(do_temp, do_unique):
        if do_temp and not do_unique:  # give latter precedence
            return "_temp_model"

        if self.logs_use_full_model_name:
            return "{}_{}vals__".format(self.model_name, self._times_validated)
        else:
            return "max_{:.3f}_{}vals__".format(self.best_key_metric,
                                                self._times_validated)

    def _save(basepath, overwrite):
        def _maybe_save(save_fn, path, overwrite):
            def _make_unique_path(path):
                _dir = Path(path).parent
                stem = Path(path).stem
                ext = Path(path).suffix

                existing_names = [x.name for x in Path(_dir).iterdir()
                                  if x.is_file()]
                new_name = stem + '_v2' + ext
                i = 2
                while new_name in existing_names:
                    i += 1
                    new_name = stem + f'_v{i}' + ext
                return os.path.join(_dir, new_name)

            file_exists = Path(path).is_file()
            if not file_exists or (file_exists and overwrite):
                save_fn(path)
            elif file_exists and not overwrite:
                path = _make_unique_path(path)
                save_fn(path)

        if overwrite not in (None, True, False):
            raise ValueError("`overwrite` must be one of: None, True, False")
        if overwrite is None:
            overwrite = bool(self.checkpoints_overwrite_duplicates)

        save_fns = [(basepath + 'state.h5',   self.save),
                    (basepath + 'hist.png',   self._save_history_fig),
                    (basepath + 'report.png', self.save_report)]

        _sf = self._make_model_save_fns(basepath)
        save_fns.extend(_sf)

        for path, save_fn in save_fns:
            _maybe_save(save_fn, path, overwrite)

    def _clear_checkpoints_IF():
        def _filter_varying(string):
            """Omit changing chars to infer uniques per checkpoint"""
            # omit digits, which change across `max`, `vals`, etc
            filtered = ''.join(s for s in string if not s.isdigit())

            # omit versions, e.g. _v2, _v3, introduced w/ overwrite=True
            stem, ext = Path(filtered).stem, Path(filtered).suffix
            if stem[-2:] == '_v':  # digit already filtered
                stem = stem[:-2]
            return stem + ext

        paths = [f for f in Path(self.logdir).iterdir() if f.is_file()]
        files_per_checkpoint = len(set(_filter_varying(p.name) for p in paths))
        paths = sorted(paths, key=os.path.getmtime)
        paths = list(map(str, paths))

        # compare directly against `paths` to avoid over-deletion
        while len(paths) / files_per_checkpoint > max(1, self.max_checkpoints):
            # remove oldest first (by creation time)
            [os.remove(paths.pop(0)) for _ in range(files_per_checkpoint)]

    do_temp = self._should_do(self.temp_checkpoint_freq)
    do_unique = self._should_do(self.unique_checkpoint_freq)
    if not (do_temp or do_unique) and not forced:
        return

    # to keep savename accurate in `logs_use_full_model_name` case
    _update_best_key_metric_in_model_name(self)

    savename = _get_savename(do_temp, do_unique)
    basepath = os.path.join(self.logdir, savename)
    _save(basepath, overwrite)

    try:
        _clear_checkpoints_IF()
    except BaseException as e:
        print(WARN, "Checkpoint files could not be cleared; skipping\n"
              "Errmsg:", e)


def _make_model_save_fns(self, basepath):
    save_fns = []
    if 'model' not in self.saveskip_list:
        name = 'model'
        if not self.model_save_kw.get('include_optimizer', True):
            name += '_noopt'
        name += '.' + self.model_save_kw.get('save_format', 'h5')
        save_fns.append((
            basepath + name,
            lambda path: self.model.save(path, **self.model_save_kw)))
    if 'model:weights' not in self.saveskip_list:
        name = 'weights.' + self.model_save_weights_kw.get('save_format', 'h5')
        save_fns.append((
            basepath + name,
            lambda path: self.model.save_weights(
                path, **self.model_save_weights_kw)))
    return save_fns


def save(self, savepath=None):
    """Save `TrainGenerator` state (including both `DataGenerator`s), and if
    configured to, model weights and optimizer state. Applies callbacks with
    `stage='save'` *before* saving to file.

    Arguments:
        savepath: str / None
            File path to save to. If None, will set to
            `logdir` + `_temp_model__state.h5`. Internally, `savepath` is
            passed meaningfully by :meth:`checkpoint` and
            :meth:`_save_best_model`.

    **Saving TrainGenerator state**:

    Configured with `saveskip_list`, and `DataGenerator.saveskip_list` for
    `datagen` and `val_datagen`; any attribute *not* included in the lists will
    be saved (except objects that cannot be pickled, which will raise
    `PickleError`. Callables (e.g. functions) are excluded automatically).

    **Saving optimizer state**:

    Configured with `optimizer_save_configs`, in the below structure;
    only one of `'include'`, `'exclude'` can be set.

    >>> {'include':  # optimizer attributes to include
    ...      ['weights', 'learning_rate']  # will ONLY save these
    ... }
    >>> {'exclude':  # optimizer attributes to exclude
    ...      ['updates', 'epsilon']  # will save everything BUT these
    ... }

    **Note**:

    :meth:`checkpoint` or :meth:`_save_best_model`` called from within
    :meth:`TrainGenerator._on_val_end` will set `_save_from_on_val_end=True`,
    which will then set validation flags so as to not repeat call to
    `_on_val_end` upon loading `TrainGenerator`.
    """
    def _cache_datagen_attributes():
        """Temporarily store away `DataGenerator` attributes so that
        `datagen` and `val_datagen` can be saved directly according to
        `DataGenerator.saveskip_list`, and then restore."""
        def _cache_then_del_attrs(parent_obj, child_obj_name, to_exclude):
            cached_attrs = {}
            obj = getattr(parent_obj, child_obj_name)

            for attr_name in to_exclude:
                cache_name = child_obj_name + '.' + attr_name
                attr_value = getattr(obj, attr_name)
                cached_attrs[cache_name] = attr_value
                delattr(obj, attr_name)
            return cached_attrs

        cached_attrs = {}
        for dg_name in ('datagen', 'val_datagen'):
            dg = getattr(self, dg_name)
            to_exclude = dg.saveskip_list.copy()

            for key, val in vars(dg).items():
                # exclude callables (e.g. functions), which *might* be
                # pickleable, but no need to
                if isinstance(val, types.LambdaType):
                    to_exclude.append(key)
            cached_attrs.update(_cache_then_del_attrs(self, dg_name, to_exclude))
        return cached_attrs

    def _restore_cached_attributes(parent_obj, cached_attrs):
        for obj_attr_name in cached_attrs:
            obj_name, attr_name = obj_attr_name.split('.')
            obj = getattr(parent_obj, obj_name)
            attr_value = cached_attrs[obj_attr_name]
            setattr(obj, attr_name, attr_value)

    savepath = savepath or os.path.join(self.logdir, '_temp_model__state.h5')
    cached_attrs = _cache_datagen_attributes()

    self._apply_callbacks(stage='save')

    if self._save_from_on_val_end:
        # mark end of validation so loaded model won't repeat _on_val_end
        self._inferred_batch_size = None
        self._val_loop_done = False
        self._train_loop_done = False
        self._save_from_on_val_end = False

    skiplist = self.saveskip_list + ['model']  # do not pickle model
    savedict = {k: v for k, v in vars(self).items() if k not in skiplist}
    if 'optimizer_state' not in self.saveskip_list:
        savedict['optimizer_state'] = self._get_optimizer_state()

    try:
        with open(savepath, "wb") as savefile:
            pickle.dump(savedict, savefile)
            print("TrainGenerator state saved")
    except BaseException as e:
        print(WARN, "TrainGenerator state could not be saved; skipping...\n"
              "Errmsg:", e)
    _restore_cached_attributes(self, cached_attrs)


def _get_optimizer_state(self):
    """Get optimizer attributes to save, according to `optimizer_save_configs`;
    helper method to :meth:`save`.
    """
    def _get_attrs_to_save(opt):
        cfg = self.optimizer_save_configs
        all_attrs = exclude_unpickleable(vars(opt))
        all_attrs['weights'] = []

        if cfg is None:
            return all_attrs

        if 'exclude' in cfg:
            return [a for a in all_attrs if a not in cfg['exclude']]

        if 'include' in cfg:
            attrs = []
            for attr in cfg['include']:
                if attr in all_attrs:
                    attrs.append(attr)
                elif attr in vars(opt):
                    print(WARN, ("'{}' optimizer attribute cannot be "
                                 "pickled; skipping").format(attr))
                else:
                    print(WARN, ("'{}' attribute not found in optimizer; "
                                 "skipping").format(attr))
            return attrs

    state = {}
    opt = self.model.optimizer
    to_save = _get_attrs_to_save(opt)
    for name in to_save:
        if name == 'weights':
            weights = opt.get_weights()
            if weights != []:
                state['weights'] = weights
            continue

        value = getattr(opt, name, None)
        if isinstance(value, tf.Variable):
            state[name] = tensor_utils.eval_tensor(value, backend=K)
        else:
            state[name] = value
    return state


def load(self, filepath=None, passed_args=None):
    """Loads `TrainGenerator` state (including both `DataGenerator`s), and if
    configured to, model optimizer attributes and instantiates optimizer
    (but not model architecture). Instantiates callbacks, and applies them with
    `stage='load'`. Preloads data from `datagen` and `val_datagen`.

    Arguments:
        filepath: str / None
            File path to load from. If None:

            - `logdir` is None, or is not a directory: raises `ValueError`
            - `logdir` is a directory: sets `filepath` to *latest* file
              with name ending with `'__state.h5'`; if there isn't such file,
              raises `ValueError`.

        passed_args: dict / None
            Passed within `TrainGenerator.__init__()` as arguments given by user
            (*not* defaults) to `__init__`. Along `loadskip_list`, mediates
            which attributes are loaded (see below).

    **Loading TrainGenerator state**:

    Configured with `loadskip_list`, `DataGenerator.loadskip_list` and
    `DataGenerator.preprocessor.loadskip_list` for `datagen` and `val_datagen`;
    any attribute *not* included in the lists will be loaded. `'model'` is
    always skipped from loading as part of pickled file, since it's
    never saved via :meth:`save`.

    - If `loadskip_list == 'auto'` or falsy (e.g. None), will default it to
      `passed_args`.
    - If `passed_args` is falsy, defaults to `[]` (load all).
    - If `'{auto}'` is in `loadskip_list`, then will append `passed_args`
      to `loadskip_list`, and pop `'{auto}'`.
    - Will omit `'datagen'` & `'val_datagen'` from `passed_args`; only way to
      skip them is via `self.loadskip_list`.

    **Loading optimizer state**:

    Configured with `optimizer_load_configs`, in the below structure;
    only one of `'include'`, `'exclude'` can be set.

    >>> {'include':  # optimizer attributes to include
    ...      ['weights', 'learning_rate']  # will ONLY load these
    ... }
    >>> {'exclude':  # optimizer attributes to exclude
    ...      ['updates', 'epsilon']  # will load everything BUT these
    ... }
    """
    def _get_loadskip_list(passed_args):
        if passed_args:
            # omit datagen & val_datagen only if in `self.loadskip_list`
            del passed_args['datagen']
            del passed_args['val_datagen']
            passed_args = list(passed_args)
        else:
            passed_args = []

        if self.loadskip_list == 'auto' or not self.loadskip_list:
            return passed_args
        elif '{auto}' in self.loadskip_list:
            lsl = self.loadskip_list.copy()
            if passed_args:
                lsl += passed_args
            lsl.pop(lsl.index('{auto}'))
            return lsl
        elif self.loadskip_list == 'none':
            return []
        else:
            return self.loadskip_list

    def _get_filepath(filepath):
        filepath = filepath or self.loadpath
        if filepath is not None:
            return filepath

        if self.logdir is None:
            raise ValueError("`filepath`, `loadpath`, and `logdir` are None")
        elif not Path(self.logdir).is_dir():
            raise ValueError("`filepath` is None, and `logdir` is not a folder"
                             "(%s)" % self.logdir)

        paths = []
        for path in Path(self.logdir).iterdir():
            if path.name.endswith('__state.h5'):
                paths.append(path)

        if not paths:
            raise ValueError("`filepath` is None, and no __state.h5 files "
                             f"found in `logdir` ({self.logdir})")
        paths.sort(key=os.path.getmtime)
        return paths[-1]  # latest

    def _load_datagen_attrs(loadfile_parsed):
        def _load_preprocessor_attrs(dg, dg_loaded, dg_name):
            pp = dg.preprocessor
            pp_loaded = dg_loaded.preprocessor

            for attr, value in vars(pp_loaded).items():
                if attr not in pp.loadskip_list:
                    setattr(pp, attr, value)

        def _validate_set_nums(dg, dg_loaded):
            if hasattr(dg, 'set_nums_original'):
                if any(set_num not in dg.set_nums_original
                       for set_num in dg_loaded.set_nums_to_process):
                    print(WARN, "found set_num in loaded `set_nums_to_process` "
                          "that isn't in the passed `set_nums_original`; setting "
                          "former to `set_nums_original`.")
                    dg.set_nums_to_process = dg.set_nums_original.copy()

        dg_names = [n for n in ('datagen', 'val_datagen') if n in loadfile_parsed]

        for dg_name in dg_names:
            dg = getattr(self, dg_name)
            dg_loaded = loadfile_parsed.pop(dg_name)
            lsl_loaded = None

            for attr, value in vars(dg_loaded).items():
                if attr not in dg.loadskip_list:
                    if attr == 'set_nums_to_process':
                        _validate_set_nums(dg, dg_loaded)
                    elif attr == 'loadskip_list':
                        # delay setting since it changes iteration logic
                        lsl_loaded = value
                    else:
                        setattr(dg, attr, value)
            _load_preprocessor_attrs(dg, dg_loaded, dg_name)
            if lsl_loaded is not None:
                dg.loadskip_list = lsl_loaded

    filepath = _get_filepath(filepath)
    with open(filepath, 'rb') as loadfile:
        loadfile_parsed = pickle.load(loadfile)
        # drop items in loadskip_list (e.g. to avoid overriding passed kwargs)
        loadskip_list = _get_loadskip_list(passed_args)
        # `model` as attribute is never saved via .save()
        loadskip_list.append('model')
        for name in loadskip_list:
            loadfile_parsed.pop(name, None)

        if 'datagen' in loadfile_parsed or 'val_datagen' in loadfile_parsed:
            _load_datagen_attrs(loadfile_parsed)

        # assign loaded/cached attributes
        self.__dict__.update(loadfile_parsed)

        if 'optimizer_state' not in loadskip_list:
            if getattr(self, 'optimizer_state', None):
                _load_optimizer_state(self)
            else:
                # if `optimizer_state` wasn't to be saved, then it not loading
                # is expected
                if ('saveskip_list'   not in loadfile_parsed or
                    'optimizer_state' not in loadfile_parsed['saveskip_list']):
                    print(WARN, "'optimizer_state' not found in loadfile; "
                          "skipping. (Optimizer will still instantiate before "
                          ".train())")
                _init_optimizer(
                    self.model, self.class_weights,
                    input_as_labels=self.input_as_labels,
                    alias_to_metric_name_fn=self._alias_to_metric_name)
        print("TrainGenerator state loaded from", filepath)

    print("--Preloading excluded data based on datagen states ...")
    self._prepare_initial_data(from_load=True)
    print("... finished--")

    if not self._init_callbacks_called:
        self._init_callbacks()
    self._apply_callbacks(stage='load')


def _load_optimizer_state(self):
    """Sets optimizer attributes from `self.optimizer_state`, according to
    `optimizer_load_configs`; helper method to :meth:`load`. Is called internally
    by :meth:`load`. `optimizer_state` is set to None to free memory afterwards.
    """
    def _get_attrs_to_load(opt):
        cfg = self.optimizer_load_configs
        all_attrs = [a for a in list(vars(opt)) if a != 'updates']

        if cfg is None:
            return all_attrs
        if 'exclude' in cfg:
            return [a for a in all_attrs if a not in cfg['exclude']]
        if 'include' in cfg:
            return [a for a in all_attrs if a in cfg['include']]

    opt = self.model.optimizer
    to_load = _get_attrs_to_load(opt)

    for name, value in self.optimizer_state.items():
        if name not in to_load:
            continue
        elif name == 'weights':
            continue  # set later

        if isinstance(getattr(opt, name), tf.Variable):
            K.set_value(getattr(opt, name), value)
        else:
            setattr(opt, name, value)

    _init_optimizer(self.model, self.class_weights,
                    input_as_labels=self.input_as_labels,
                    alias_to_metric_name_fn=self._alias_to_metric_name)
    if 'weights' in self.optimizer_state:
        self.model.optimizer.set_weights(self.optimizer_state['weights'])

    self.optimizer_state = None  # free up memory
    print("Optimizer state loaded (& cleared from TrainGenerator)")


def _save_history_fig(self, savepath=None):
    """Saves `_history_fig`. Does nothing if `_history_fig` is falsy (e.g. None).

    Arguments:
        savepath: str / None
            Path to save figure to. If None, will set to `logdir` +
            `'_temp_model__hist.png'`. Also if None, and `final_fig_dir`
            is set, will use full model name instead.

    If `final_fig_dir` is set, will also save (at most one per `model_num`)
    figure there using full model name; if there are existing savefiles (.png)
    with same `model_num`, will delete them. Intended to document latest state
    of history.
    """
    def _save_final_fig():
        prev_figs = [os.path.join(self.final_fig_dir, name)
                     for name in os.listdir(self.final_fig_dir)
                     if (name.startswith('M' + str(self.model_num)) and
                         name.endswith('.png'))]
        if len(prev_figs) != 0:
            [os.remove(fig) for fig in prev_figs]

        _savepath = os.path.join(self.final_fig_dir, self.model_name + '.png')
        self._history_fig.savefig(_savepath)

    _path = savepath or os.path.join(self.logdir, '_temp_model__hist.png')
    if self._history_fig:
        try:
            self._history_fig.savefig(_path)
        except Exception as e:
            print(WARN, "Model history could not be saved; skipping",
                  "\nErrmsg:", e)

    if self.final_fig_dir:  # keep at most one per model_num
        pass_on_error(_save_final_fig, errmsg=("Final fig could not be "
                                               "saved; skipping"))


def _update_best_key_metric_in_model_name(self):
    keyword = self.model_name_configs['best_key_metric']
    self.model_name = self.model_name.split(keyword)[0] + keyword + (
        '%.3f' % self.best_key_metric).replace('0.', '.')
