import builtins
import numpy as np
from functools import reduce
from collections.abc import Mapping


def ordered_shuffle(*args):
    """Shuffles each of the iterables the same way. Ex:

    >>> ([1, 2, 3, 4], {'a': 5, 'b': 6, 'c': 7, 'd': 8})
    >>> ([3, 4, 1, 2], {'c': 7, 'd': 8, 'a': 5, 'b': 6})
    """
    zipped_args = list(zip(*(a.items() if isinstance(a, dict)
                             else a for a in args)))
    np.random.shuffle(zipped_args)
    return [(_type(data) if _type != np.ndarray else np.asarray(data))
            for _type, data in zip(map(type, args), zip(*zipped_args))]


def nCk(n, k):
    """n-Choose-k"""
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom


def builtin_or_npscalar(x, include_type_type=False):
    """Returns True if x is a builtin or a numpy scalar. Since `type` is
    a builtin, but is a class rather than a literal, it's omitted by default;
    set `include_type_type=True` to include it.
    """
    value = isinstance(x, (np.generic, type(None), type(min))
                       ) or type(x) in vars(builtins).values()
    return value if include_type_type else (value and not isinstance(x, type))


def obj_to_str(x, len_lim=200, drop_absname=False):
    """Converts `x` to a string representation if it isn't a builtin or numpy
    scalar.

    Trims string representation to `len_lim` if `x` or `type(x)` have no
    `__qualname__` or `__name__` attributes. To drop packages and modules in
    an object's name (package.subpackage.obj), pass `drop_absname=True`.
    """
    if builtin_or_npscalar(x, include_type_type=False):
        return x
    if hasattr(x, '__qualname__') or hasattr(x, '__name__'):
        qname = getattr(x, '__qualname__', None)
        name  = getattr(x, '__name__', None)
    else:
        # fallback to class if object has no name
        qname = getattr(type(x), '__qualname__', None)
        name  = getattr(type(x), '__name__', None)

    out = qname or name
    if not out:
        # fallback to str or repr if still no name
        out = str(x) if hasattr(x, '__str__') else repr(x)
        out = out[:len_lim]
    elif drop_absname:
        out = out.split('.')[-1]
    return out


def deeplen(item):
    """Return total number of items in an arbitrarily nested iterable - excluding
    the iterables themselves."""
    if isinstance(item, np.ndarray):
        return item.size
    try:
        list(iter(item))
    except:
        return 1
    if isinstance(item, str):
        return 1
    if isinstance(item, Mapping):
        item = item.values()
    return sum(deeplen(subitem) for subitem in item)


def deepget(obj, key=None, drop_keys=0):
    """Get an item from an arbitarily nested iterable. `key` is a list/tuple of
    indices of access specifiers (indices or mapping (e.g. dict) keys); if a
    mapping is unordered (e.g. dict for Python <=3.5), retrieval isn't consistent.
    """
    if not key or not obj:
        return obj
    if drop_keys != 0:
        key = key[:-drop_keys]
    for k in key:
        if isinstance(obj, Mapping):
            k = list(obj)[k]  # get key by index (OrderedDict, Python >=3.6)
        obj = obj[k]
    return obj


def deepmap(obj, fn):
    """Map `fn` to items of an arbitrarily nested iterable, *including* iterables.
    See https://codereview.stackexchange.com/q/242369/210581 for an explanation.
    """
    def dkey(x, k):
        return list(x)[k] if isinstance(x, Mapping) else k

    def nonempty_iter(item):
        # do not enter empty iterable, since nothing to 'iterate' or apply fn to
        try:
            list(iter(item))
        except:
            return False
        return not isinstance(item, str) and len(item) > 0

    def _process_key(obj, key, depth, revert_tuple_keys, recursive=False):
        container = deepget(obj, key, 1)
        item      = deepget(obj, key, 0)

        if nonempty_iter(item) and not recursive:
            depth += 1
        if len(key) == depth:
            if key[-1] == len(container) - 1:  # iterable end reached
                depth -= 1      # exit iterable
                key = key[:-1]  # drop iterable key
                if key in revert_tuple_keys:
                    supercontainer = deepget(obj, key, 1)
                    k = dkey(supercontainer, key[-1])
                    supercontainer[k] = tuple(deepget(obj, key))
                    revert_tuple_keys.pop(revert_tuple_keys.index(key))
                if depth == 0 or len(key) == 0:
                    key = None  # exit flag
                else:
                    # recursively exit iterables, decrementing depth
                    # and dropping last key with each recursion
                    key, depth = _process_key(obj, key, depth, revert_tuple_keys,
                                              recursive=True)
            else:  # iterate next element
                key[-1] += 1
        elif depth > len(key):
            key.append(0)  # iterable entry
        return key, depth

    key = [0]
    depth = 1
    revert_tuple_keys = []

    if not nonempty_iter(obj):  # nothing to do here
        raise ValueError(f"input must be a non-empty iterable - got: {obj}")
    if isinstance(obj, tuple):
        obj = list(obj)
        revert_tuple_keys.append(None)  # revert to tuple at function exit

    while key is not None:
        container = deepget(obj, key, 1)
        item      = deepget(obj, key, 0)

        if isinstance(container, tuple):
            ls = list(container)  # cast to list to enable mutating
            ls[key[-1]] = fn(item, key)

            supercontainer = deepget(obj, key, 2)
            k = dkey(supercontainer, key[-2])
            supercontainer[k] = ls
            revert_tuple_keys.append(key[:-1])  # revert to tuple at iterable exit
        else:
            k = dkey(container, key[-1])
            container[k] = fn(item, key)

        key, depth = _process_key(obj, key, depth, revert_tuple_keys)

    if None in revert_tuple_keys:
        obj = tuple(obj)
    return obj


def deep_isinstance(obj, cond):
    """Checks that items within an arbitrarily nested iterable meet `cond`.
    Returns a list of bools; to assert that *all* elements meet `cond`, run
    `all(deep_isinstance())`.
    """
    bools = []
    def fn(item, key=None):
        if isinstance(item, str):
            bools.append(cond(item))
            return item
        try:
            list(iter(item))
        except TypeError:
            bools.append(cond(item))
        return item

    try:
        list(iter(obj))
        if not len(obj) > 0:
            raise Exception
        deepmap(obj, fn)
    except:
        fn(obj)
    return bools
