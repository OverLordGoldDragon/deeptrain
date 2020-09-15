from pprint import pprint
from collections.abc import Mapping
from .algorithms import deepget, deepmap, deep_isinstance
from .algorithms import builtin_or_npscalar


def deepcopy_v2(obj, item_fn=None, skip_flag=42069, debug_verbose=False):
    """Enables customized copying of a nested iterable, mediated by `item_fn`."""
    if item_fn is None:
        item_fn = lambda item: item
    copied = [] if isinstance(obj, (list, tuple)) else {}
    copied_key = []
    revert_tuple_keys = []
    copy_paused = [False]
    key_decrements = [0]
    skipref_key = []

    def dkey(x, k):
        return list(x)[k] if isinstance(x, Mapping) else k

    def isiter(item):
        try:
            list(iter(item))
            return not isinstance(item, str)
        except:
            return False

    def reconstruct(item, key):
        def _container_or_elem(item):
            if isiter(item):
                if isinstance(item, (tuple, list)):
                    return []
                elif isinstance(item, Mapping):
                    return {}
            return item_fn(item)

        def _obj_key_advanced(key, skipref_key):
            # [1, 1]    [1, 0] -> True
            # [2]       [1, 0] -> True
            # [2, 0]    [1, 0] -> True
            # [1, 0]    [1, 0] -> False
            # [1]       [1, 0] -> False
            # [1, 0, 1] [1, 0] -> False
            i = 0
            while (i < len(key) - 1 and i < len(skipref_key) - 1) and (
                    key[i] == skipref_key[i]):
                i += 1
            return key[i] > skipref_key[i]

        def _update_copied_key(key, copied_key, on_skip=False):
            ck = []
            for k, k_decrement in zip(key, key_decrements):
                ck.append(k - k_decrement)
            copied_key[:] = ck

        def _update_key_decrements(key, key_decrements, on_skip=False):
            if on_skip:
                while len(key_decrements) < len(key):
                    key_decrements.append(0)
                while len(key_decrements) > len(key):
                    key_decrements.pop()
                key_decrements[len(key) - 1] += 1
            else:
                while len(key_decrements) < len(key):
                    key_decrements.append(0)
                while len(key_decrements) > len(key):
                    key_decrements.pop()

        def _copy(obj, copied, key, copied_key, _item):
            container = deepget(copied, copied_key, 1)

            if isinstance(container, list):
                container.insert(copied_key[-1], _item)
            elif isinstance(container, str):
                # str container implies container was transformed to str by
                # item_fn; continue skipping until deepmap exits container in obj
                pass
            else:  # tuple will yield error, no need to catch
                obj_container = deepget(obj, key, 1)
                k = dkey(obj_container, key[-1])
                if debug_verbose:
                    print("OBJ_CONTAINER:", obj_container, key[-1])
                    print("CONTAINER:", container, k, '\n')
                container[k] = _item

        if copy_paused[0] and not _obj_key_advanced(key, skipref_key):
            if debug_verbose:
                print(">SKIP:", item)
            return item


        _item = _container_or_elem(item)
        if isinstance(_item, int) and _item == skip_flag:
            copy_paused[0] = True
            _update_key_decrements(key, key_decrements, on_skip=True)
            skipref_key[:] = key
            if debug_verbose:
                print("SKIP:", key, key_decrements, copied_key)
                print(item)
            return item
        copy_paused[0] = False

        _update_key_decrements(key, key_decrements)
        while len(key_decrements) > len(key):
            key_decrements.pop()
            if debug_verbose:
                pprint("POP: {} {} {}".format(key, key_decrements, copied_key))
        _update_copied_key(key, copied_key)

        if debug_verbose:
            print("\nSTUFF:", key, key_decrements, copied_key, len(copied))
            print(_item)
            for k, v in copied.items():
                print(k, '--', v)
            print()
        _copy(obj, copied, key, copied_key, _item)
        if debug_verbose:
            print("###########################################################",
                  len(copied))

        if isinstance(item, tuple):
            revert_tuple_keys.append(copied_key.copy())
        return item

    def _revert_tuples(copied, obj, revert_tuple_keys):
        revert_tuple_keys = list(reversed(sorted(revert_tuple_keys, key=len)))
        for key in revert_tuple_keys:
            supercontainer = deepget(copied, key, 1)
            container      = deepget(copied, key, 0)
            k = dkey(supercontainer, key[-1])
            supercontainer[k] = tuple(container)
        if isinstance(obj, tuple):
            copied = tuple(copied)
        return copied

    deepmap(obj, reconstruct)
    copied = _revert_tuples(copied, obj, revert_tuple_keys)
    return copied


def extract_pickleable(obj, skip_flag=42069):
    """Given an arbitrarily nested dict / mapping, make its copy containing only
    objects that can be pickled. Excludes functions and class instances, even
    though most such can be pickled (# TODO).
    Utilizes :func:`deepcopy_v2`.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"input must be a Mapping (dict, etc) - got: {obj}")

    def item_fn(item):
        if builtin_or_npscalar(item, include_type_type=True):
            return item
        return skip_flag
    return deepcopy_v2(obj, item_fn, skip_flag)


def exclude_unpickleable(obj):
    """Given an arbitrarily nested dict / mapping, make its copy containing only
    objects that can be pickled. Excludes functions and class instances, even
    though most such can be pickled (# TODO).
    Utilizes :func:`~deeptrain.util.algorithms.deep_isinstance`.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"input must be a Mapping (dict, etc) - got: {obj}")

    can_pickle = lambda x: builtin_or_npscalar(x, include_type_type=True)
    pickleable = {}
    for k, v in obj.items():
        bools = deep_isinstance(v, cond=can_pickle)
        if bools and all(bools):
            pickleable[k] = v
    return pickleable
