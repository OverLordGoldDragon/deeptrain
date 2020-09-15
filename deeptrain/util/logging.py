# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import textwrap
import json

from inspect import getsource
from pathlib import Path
from .algorithms import deeplen, builtin_or_npscalar
from .misc import _dict_filter_keys
from ._backend import NOTE, WARN, Image, ImageDraw, ImageFont


def save_report(self, savepath=None):
    """Saves model, `TrainGenerator`, `'datagen'`, and `'val_datagen'`
    attributes and values as an image-text, text generated with
    :meth:`generate_report`.

    Text font is set from `TrainGenerator.report_fontpath`, which defaults
    to a programming-style *consolas* for consistent vertical and horizontal
    alignment.

    If `savepath` is None, will save a temp report in `TrainGenerator.logdir`
    + `'_temp_model__report.png'`.
    """
    def _write_text_image(text, savepath, fontpath, fontsize=15,
                          width=1, height=1):
        img = Image.new('RGB', color='white',
                        size=(int(700 * width), int(300 * height)))
        fnt = ImageFont.truetype(fontpath, fontsize)

        d = ImageDraw.Draw(img)
        d.text((10,30), text, fill=(0, 0, 0), font=fnt)

        img.save(savepath)

    text = generate_report(self)
    longest_line = max(map(len, text.split('\n')))
    num_newlines = len(text.split('\n'))

    savepath = savepath or os.path.join(self.logdir, '_temp_model__report.png')
    try:
        _write_text_image(text, savepath, self.report_fontpath,
                          width=longest_line / 80,
                          height=num_newlines / 16)
        print("Model report generated and saved")
    except Exception as e:
        print(WARN, "Report could not be generated; skipping")
        print("Errmsg:", e)


def generate_report(self):
    """Generates `model`, :class:`TrainGenerator`, `datagen`, and `val_datagen`
    reports according to `report_configs`.

    Writes attributes and values in three columns of text, converted to and saved
    as an image. Extracts information from `model_configs` and `vars()` of
    `TrainGenerator`, `datagen`, and `val_datagen`. Useful for snapshotting
    key model, training, and data attributes for quick reference.

    `report_configs` is structured as follows:

    >>> {target_0:
    ...     {filter_spec_0:
    ...          [attr0, attr1, ...],
    ...     },
    ...     {filter_spec_1:
    ...          [attr0, attr1, ...],
    ...     },
    ... }

    - `target` is one of: `'model', 'traingen', 'datagen', 'val_datagen'`;
      it may be a tuple of, in which case `filter_spec` applies to those included.
    - `filter_spec` is one of: `'include', 'exclude', 'exclude_types'`, but cannot
      include `'include'` and `'exclude'` at once.

      - `'include'`: names of attributes to include in report; no other attribute
        will be included. Supports wildcards with `'*'` as leading character;
        e.g. `'*_has_'` will include all names starting with _has_.
      - `'exclude'`: names of attributes to exclude from report; all other
        attributes will be included. Also supports wildcards.
      - `'exclude_types'`: attribute types to exclude from report. Elements
        of this list cannot be string, unless prepended with `'#'`, which
        specifies an exception. E.g. if `attr` is dict, and `dict` is in the list,
        then it can be kept in report by including `'#attr'` in the list.

    See :data:`~deeptrain.util._default_configs._DEFAULT_REPORT_CFG` for the
    default `report_configs`, containing every possible config case.
    """
    # TODO: lists are =-separated
    def _list_to_str_side_by_side_by_side(_list, space_between_cols=0):
        def _split_in_three(_list):
            def _pad_column_bottom(_list):
                l = len(_list) // 3
                to_fill = 3 - len(_list) % 3

                if to_fill == 1:
                    _list.insert(l, '')
                elif to_fill == 2:
                    _list.insert(l, '')
                    _list.insert(2 * l + 1, '')
                return _list

            L = len(_list) / 3
            if not L.is_integer():
                # L must be integer to preserve all rows
                _list = _pad_column_bottom(_list)
            L = len(_list) // 3

            return _list[:L], _list[L:2*L], _list[2*L:]

        def _exclude_chars(_str, chars):
            return ''.join([c for c in _str if c not in chars])

        list1, list2, list3 = _split_in_three(_list)
        longest_str1 = max(map(len, map(str, list1)))
        longest_str2 = max(map(len, map(str, list2)))

        _str = ''
        for entries in zip(list1, list2, list3):
            left, mid, right = [_exclude_chars(str(x), '[]') for x in entries]
            left += " " * (longest_str1 - len(left) + space_between_cols)
            mid  += " " * (longest_str2 - len(mid)  + space_between_cols)

            _str += left + mid + right + '\n'
        return _str

    def _process_attributes_to_text_dicts():
        def _validate_report_configs(cfg):
            def _validate_keys(keys):
                supported = ('model', 'traingen', 'datagen', 'val_datagen')
                for key in keys:
                    if key not in supported:
                        print(WARN, "'%s' report_configs key not " % key
                              + "supported, and will be ignored; supported "
                              "are: {}".format(', '.join(supported)))
                        keys.pop(keys.index(key))
                return keys

            def _validate_subkeys(cfg):
                supported = ('include', 'exclude', 'exclude_types')
                for key, val in cfg.items():
                    if 'include' in val and 'exclude' in val:
                        raise ValueError("cannot have both 'include' and "
                                         "'exclude' subkeys in report_configs")
                    for subkey, attrs in val.items():
                        if not isinstance(attrs, list):
                            raise ValueError("report_configs subkey values must "
                                             "be lists (e.g. 'exclude' values)")
                        if subkey not in supported:
                            raise ValueError(
                                ("'{}' report_configs subkey not supported; must "
                                 "be one of: {}").format(
                                     subkey, ', '.join(supported)))
                return cfg

            def _unpack_tuple_keys(_dict):
                # e.g. ('datagen', 'val_datagen'): [*] ->
                #       'datagen': [*], 'val_datagen': [*]
                newdict = {}
                for key, val in _dict.items():
                    if isinstance(key, tuple):
                        for k in key:
                            newdict[k] = val
                    else:
                        newdict[key] = val
                return newdict

            keys = []
            for key in cfg:
                keys.extend([key] if isinstance(key, str) else key)
            keys = _validate_keys(keys)

            cfg = _unpack_tuple_keys(self.report_configs)
            cfg = {k: v for k, v in cfg.items() if k in keys}
            cfg = _validate_subkeys(cfg)

            return cfg

        def _process_wildcards(txt_dict, obj_dict, obj_cfg, exclude):
            for attr in obj_cfg:
                if attr[0] == '*':
                    from_wildcard = _dict_filter_keys(obj_dict, attr[1:],
                                                      exclude=False,
                                                      filter_substr=True).keys()
                    for key in from_wildcard:
                        if exclude and key in txt_dict:
                            del txt_dict[key]
                        elif not exclude:
                            txt_dict[key] = obj_dict[key]
            return txt_dict

        def _exclude_types(txt_dict, name, exclude_types):
            def _dict_filter_value_types(dc, types):
                if not isinstance(types, tuple):
                    types = tuple(types) if isinstance(types, list) else (types,)
                return {key: val for key, val in dc.items()
                        if not isinstance(val, types)}

            cache, types = {}, []
            for _type in exclude_types:
                if not isinstance(_type, str):
                    types.append(_type)
                elif _type[0] == '#':
                    cache[_type[1:]] = txt_dict[_type[1:]]
                else:
                    print(WARN,  "str type in 'exclude_types' of `report_configs`"
                          " is unsupported (unless as an exception specifier"
                          " w/ '#' prepended), and will be skipped"
                          " (recieved '%s')" % _type)

            txt_dict = _dict_filter_value_types(txt_dict, types)
            for attr in cache:
                txt_dict[attr] = cache[attr]  # restore cached
            return txt_dict

        cfg = _validate_report_configs(self.report_configs)

        txt_dicts = dict(model={}, traingen={}, datagen={}, val_datagen={})
        obj_dicts = (self.model_configs,
                     *map(vars, (self, self.datagen, self.val_datagen)))

        for name, obj_dict in zip(txt_dicts, obj_dicts):
            if obj_dict is not None and (name not in cfg or not cfg[name]):
                txt_dicts[name] = obj_dict
            elif obj_dict is not None:
                for subkey in cfg[name]:
                    obj_cfg = cfg[name][subkey]
                    if subkey != 'exclude_types':
                        exclude = True if subkey == 'exclude' else False
                        txt_dicts[name] = _dict_filter_keys(
                            obj_dict, obj_cfg, exclude=exclude)
                        txt_dicts[name] = _process_wildcards(
                            txt_dicts[name], obj_dict, obj_cfg, exclude)
                    else:
                        txt_dicts[name] = _exclude_types(
                            txt_dicts[name], name, obj_cfg)
        return txt_dicts

    def _postprocess_text_dicts(txt_dicts):
        def _wrap_if_long(dicts_list, len_th=80):
            for i, entry in enumerate(dicts_list):
                if len(entry) == 2 and len(str(entry[1])) > len_th:
                    dicts_list[i] = [entry[0], []]
                    wrapped = textwrap.wrap(str(entry[1]), width=len_th)
                    for line in reversed(wrapped):
                        dicts_list.insert(i + 1, [line])
            return dicts_list

        all_txt = txt_dicts.pop('model')
        for _dict in txt_dicts.values():
            all_txt += [''] + _dict

        _all_txt = _wrap_if_long(all_txt, len_th=80)
        _all_txt = _list_to_str_side_by_side_by_side(_all_txt,
                                                     space_between_cols=0)
        _all_txt = _all_txt.replace("',", "' =" ).replace("0, ", "0," ).replace(
                                    "000,", "k,").replace("000)", "k)")
        return _all_txt

    def _dict_lists_to_tuples(_dict):
        return {key: tuple(val) for key, val in _dict.items()
                if isinstance(val, list)}

    txt_dicts = _process_attributes_to_text_dicts()

    titles = (">>HYPERPARAMETERS", ">>TRAINGEN STATE", ">>TRAIN DATAGEN STATE",
              ">>VAL DATAGEN STATE")
    for (name, _dict), title in zip(txt_dicts.items(), titles):
        txt_dicts[name] = _dict_lists_to_tuples(_dict)
        txt_dicts[name] = list(map(list, _dict.items()))
        txt_dicts[name].insert(0, title)

    _all_txt = _postprocess_text_dicts(txt_dicts)
    return _all_txt


def get_unique_model_name(self, set_model_num=True):
    """Returns a unique model name, prepended by
    `f"M{model_num}__{model_base_name}"`. If `set_model_num`, also sets
    `model_num`.

    Name is generated by extracting info from `model_configs` according to
    `model_name_configs` (insertion-order sensitive), `name_process_key_fn`, and
    `new_model_num`.

    `name_process_key_fn(key, alias, attrs)`: returns a string representation of
    `key` `TrainGenerator` attribute or `model_configs` key and its value. Below
    is a description of the default function,
    :func:`~deeptrain.util._default_configs._DEFAULT_NAME_PROCESS_KEY_FN`,
    but custom implementations are supported:

        - `key`: name of attribute (and its value) to encode. Can get attribute
          of `TrainGenerator` object via `'.'`, e.g. `'datagen.batch_size'`.
        - `alias`: replaces `key` if not None
        - `attrs`: dict of object attribute-value pairs, where "object" is
          either `TrainGenerator` or an object that is its attribute.

    **Example:** (using default `name_process_key_fn`)

    >>> model_base_name == "AutoEncoder"
    >>> model_num == 8
    >>> model_name_configs == {"datagen.batch_size": "BS",
    ...                        "filters":            None,
    ...                        "optimizer":          "",
    ...                        "lr":                 "",
    ...                        "best_key_metric":    "__max"}
    >>> model_configs = {"conv_filters": [32, 64],
    ...                  "lr":           0.0002,
    ...                  "optimizer":    tf.keras.optimizers.SGD}
    >>> tg.best_key_metric == .97512  # TrainGenerator
    >>> tg.datagen.batch_size == 32
    ...
    ... # will yield
    >>> "M8__AutoEncoder-BS32-filters32_64-SGD-2e-4__max.975"

    Note that if `new_model_num` is True, then will set to +1
    the max number after `"M"` for directory names in `logs_dir`; e.g. if
    such a directory is `"M15__Classifier"`, will use `"M16"`, and set
    `model_num = 16`.

    If an object is passed to `model_configs`, its `.__name__` will be used
    as "value"; if this attribute is missing, will raise exception (default fn).

    Note that "unique" doesn't mean yielding a new name with each call to the
    function; for a name to be new, either a directory as described should have
    a higher `"M{num}"`, or other sources of information must change values
    (e.g. `TrainGenerator` attributes, like `best_key_metric`).
    """
    def _get_model_num():
        dirnames = ['M0']
        if self.logs_dir is not None:
            dirnames = [f.name for f in Path(self.logs_dir).iterdir()
                        if f.is_dir() and f.name.startswith("M")]
        if self.new_model_num:
            if len(dirnames) != 0:
                model_num = np.max([int(name.split('__')[0].replace('M', ''))
                                    for name in dirnames ]) + 1
            else:
                print(NOTE, "no existing models detected in",
                      self.logs_dir + "; starting model_num from '0'")

        if not self.new_model_num or len(dirnames) == 0:
            model_num = 0; _name='M0'
            while any((_name in filename) for filename in
                      os.listdir(self.logs_dir)):
                model_num += 1
                _name = 'M%s' % model_num
        return model_num

    model_num = _get_model_num()
    model_name = "M{}__{}".format(model_num, self.model_base_name)
    if set_model_num:
        self.model_num = model_num

    if self.model_name_configs:
        attrs = self.__dict__.copy()  # top-level shallow copy
        if self.model_configs:
            attrs.update(self.model_configs.copy())  # precedence to model_configs

        for key, alias in self.model_name_configs.items():
            if '.' in key:
                obj_name, attr = key.split('.')
                obj_attrs = vars(attrs[obj_name])
                if key[1] in obj_attrs:
                    model_name += self.name_process_key_fn(attr, alias, obj_attrs)
            elif key in attrs:
                model_name += self.name_process_key_fn(key, alias, attrs)
    return model_name


def get_last_log(self, name, best=False):
    """Returns latest savefile path from `logdir` (`best=False`) or
    `best_models_dir` (`best=True`).

    `name` is one of: `'report', 'state', 'weights', 'history', 'init_state'`.
    `'init_state'` ignores `best` (uses `=False`).
    """
    if name not in {'report', 'state', 'weights', 'history', 'init_state'}:
        raise ValueError("input must be one of 'report', 'state', 'weights', "
                         "'history'.")
    if name == 'init_state':
        return os.path.join(self.logdir, 'misc', 'init_state.json')

    _dir = self.best_models_dir if best else self.logdir

    paths = [str(p) for p in Path(_dir).iterdir()
             if (p.is_file() and p.stem.endswith('__' + name))]
    if len(paths) == 0:
        raise Exception(f"no {name} files found in {_dir}")

    paths.sort(key=os.path.getmtime, reverse=True)  # newest first
    return paths[0]


def _log_init_state(self, kwargs={}, source_lognames='__main__', savedir=None,
                    to_exclude=[], verbose=0):
    """Extract `self.__dict__` key-value pairs as string, ignoring funcs/methods
    or getting their source codes. May include kwargs passed to `__init__` via
    `kwargs`, and execution script's source code via `'__main__'` in
    `source_lognames`.

    Arguments:
        kwargs: dict
            kwargs passed to `self`'s `__init__`, in case they weren't set to
            `self` or were changed later.
        source_lognames: list[str] / str
            Names of self method attributes to get source code of. If includes
            '__main__', will get source code of execution script.
        savedir: str.
            Path to directory where to save logs. Saves a .json of `self`
            dict, and .txt of source codes (if any).
        to_exclude: list[str] / str
            Names of attributes to exclude from logging.
        verbose: bool / int[bool]
            Print save messages if successful.
    """
    def _save_logs(state, source, savedir, verbose):
        path = os.path.join(savedir, "init_state.json")
        j = json.dumps(state, indent=4)
        with open(path, 'w') as f:
            print(j, file=f)
        if verbose:
            print(str(self), "initial state saved to", path)

        if source != '':
            path = os.path.join(savedir, "init_source.txt")
            with open(path, 'w') as f:
                f.write(source)
            if verbose:
                print(str(self), "source codes saved to", path)

    def _name(x):
        if hasattr(x, '__name__'):
            return x.__name__
        elif hasattr(x, '__class__'):
            return str(x.__class__).replace("<class ", '').replace(
                ">", '').replace("'", '')
        else:
            return str(x)

    def _filter_objects(state_full, to_exclude):
        state = {}
        for k, v in state_full.items():
            if k in to_exclude:
                continue
            elif builtin_or_npscalar(v):
                if hasattr(v, '__len__') and deeplen(v) > 50:
                    v = _name(v)
                state[k] = str(v)
            else:
                state[k] = _name(v)
        return state

    def _get_source_code(state_full, source_lognames):
        def _get_main_source():
            if not hasattr(sys.modules['__main__'], '__file__'):
                return '', ''
            path = os.path.abspath(sys.modules['__main__'].__file__)
            with open(path, 'r') as f:
                return f.read(), path

        def _get_all_sourceable(keys, source, state_full):
            def not_func(x):
                return getattr(getattr(x, '__class__', None),
                               '__name__', '') not in ('function', 'method')

            to_skip = ['__main__']
            if not isinstance(keys, (list, tuple)):
                keys = [keys]

            for k in keys:
                if k in to_skip:
                    continue
                elif k not in state_full:
                    print(WARN, f"{k} not found in self.__dict__ - will skip")
                    continue
                v = state_full[k]
                if (not builtin_or_npscalar(v) and
                    not isinstance(v, np.ndarray) and
                    not_func(v)):
                    v = v.__class__
                try:
                    source[_name(v)] = getsource(v)
                except Exception as e:
                    print("Failed to log:", k, v, "-- skipping. "
                          "Errmsg: %s" % e)
            return source

        def _to_text(source):
            def _wrap_decor(x):
                """Format as: ## long_text_s ##
                              ## tuff #########"""
                wrapped = textwrap.wrap(x, width=77)
                txt = ''
                for line in wrapped:
                    txt += "## %s\n" % (line + ' ' + "#" * 77)[:80]
                return txt.rstrip('\n')

            txt = ''
            for k, v in source.items():
                txt += "\n\n{}\n{}".format(_wrap_decor(k), v)
            return txt.lstrip('\n')

        source = {}
        if source_lognames == '*':
            source_lognames = list(state_full)
        source = _get_all_sourceable(source_lognames, source, state_full)

        if '__main__' in source_lognames or source_lognames == '*':
            src, path = _get_main_source()
            source[path] = src

        source = _to_text(source)
        return source

    if not isinstance(to_exclude, (list, tuple)):
        to_exclude = [to_exclude]

    state_full = vars(self)
    for k, v in kwargs.items():
        if k not in state_full:
            state_full[k] = v
    state = _filter_objects(state_full, to_exclude)

    if source_lognames is not None:
        source = _get_source_code(state_full, source_lognames)
    else:
        source = ''

    if savedir is not None:
        _save_logs(state, source, savedir, verbose)
    return state, source
