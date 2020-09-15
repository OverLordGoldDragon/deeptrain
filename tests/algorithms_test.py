# -*- coding: utf-8 -*-
import os
import sys
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(__file__)
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

import pytest
import numpy as np

from collections.abc import Iterable
from copy import deepcopy
from time import time

from backend import notify, _get_test_names
from deeptrain.util.algorithms import deeplen, deepmap, obj_to_str
from deeptrain.util.algorithms import nCk, ordered_shuffle
from deeptrain.util.experimental import deepcopy_v2


tests_done = {}


def _make_bignest():
    arrays = [np.random.randn(100, 100), np.random.uniform(30, 40, 10)]
    lists = [[1, 2, '3', '4', 5, [6, 7]] * 555, {'a': 1, 'b': arrays[0]}]
    dicts = {'x': [1, {2: [3, 4]}, [5, '6', {'7': 8}]*99] * 55,
             'b': [{'a': 5, 'b': 3}] * 333, ('k', 'g'): (5, 9, [1, 2])}
    tuples = (1, (2, {3: np.array([4., 5.])}, (6, 7, 8, 9) * 21) * 99,
              (10, (11,) * 5) * 666)
    return {'arrays': arrays, 'lists': lists,
            'dicts': dicts, 'tuples': tuples}


@notify(tests_done)
def test_nCk():
    assert nCk(10, 2) == 45
    assert nCk(4, 5) == 1


@notify(tests_done)
def test_ordered_shuffle():
    ls = [1, 2, 3, 4, 'a']
    x = np.array([5, 6, 7, 8, 9])
    dc = {'a': 1, 5: ls, (2, 3): x, '4': None, None: {1: 2}}
    ls, x, dc = ordered_shuffle(ls, x, dc)

    assert len(ls) == len(x) == len(dc) == 5
    assert isinstance(ls, list)
    assert isinstance(x, np.ndarray)
    assert isinstance(dc, dict)


@notify(tests_done)
def test_deeplen():
    def _print_report(bignest, t0):
        t = time() - t0
        print("{:.5f} / iter ({} iter avg, total time: {:.3f}); sizes:".format(
            t / iters, iters, t))
        print("bignest:", deeplen(bignest))
        print(("{} {}\n" * len(bignest)).format(
            *[x for k, v in bignest.items()
              for x in ((k + ':').ljust(8), deeplen(v))]))

    iters = 2
    bignest = _make_bignest()
    t0 = time()
    for _ in range(iters):
        deeplen(bignest)
    _print_report(bignest, t0)


@notify(tests_done)
def test_deepmap():
    def fn1(x, key):
        return str(x) if not isinstance(x, Iterable) else x

    def fn2(x, key):
        return x ** 2 if isinstance(x, (int, float, np.generic)) else x

    def fn3(x, key):
        return str(x)

    def deeplen(obj):
        count = [0]
        def fn(x, key):
            if not isinstance(x, Iterable) or isinstance(x, str):
                count[0] += 1
            return x
        deepmap(obj, fn)
        return count[0]

    #### CORRECTNESS  #########################################################
    np.random.seed(4)
    arr = np.random.randint(0, 9, (2, 2))
    obj = (1, {'a': 3, 'b': 4, 'c': ('5', 6., (7, 8)), 'd': 9}, {}, arr)

    out1 = deepmap(deepcopy(obj), fn1)
    assert str(out1) == ("('1', {'a': '3', 'b': '4', 'c': ('5', '6.0', ('7', '8'))"
                         ", 'd': '9'}, {}, array([[7, 5],\n       [1, 8]]))")
    out2 = deepmap(deepcopy(obj), fn2)
    assert str(out2) == ("(1, {'a': 9, 'b': 16, 'c': ('5', 36.0, (49, 64)), "
                         "'d': 81}, {}, array([[49, 25],\n       [ 1, 64]]))")
    out3 = deepmap(deepcopy(obj), fn3)
    assert str(out3) == (r"""('1', "{'a': 3, 'b': 4, 'c': ('5', 6.0, (7, 8)), """
                         r"""'d': 9}", '{}', '[[7 5]\n [1 8]]')""")
    try:
        deepmap([], fn1)
    except ValueError:
        pass
    except:
        print("Failed to catch invalid input")

    #### PERFORMANCE  #########################################################
    bigobj  = _make_bignest()

    _bigobj = deepcopy(bigobj)
    t0 = time()
    assert deeplen(bigobj) == 53676
    print("deeplen:     {:.3f} sec".format(time() - t0))
    assert str(bigobj) == str(_bigobj)  # deeplen should not mutate `bigobj`

    bigobj = deepcopy(_bigobj)
    t0 = time()
    deepmap(bigobj, fn1)
    print("deepmap-fn1: {:.3f} sec".format(time() - t0))

    # deepmap-fn2 takes too long


@notify(tests_done)
def test_deepcopy_v2():
    def _obj_to_str(x):
        return obj_to_str(x, drop_absname=True
                          ) if not isinstance(x, np.ndarray) else str(x)

    class Dummy():
        pass

    np.random.seed(4)
    arr = np.random.randint(0, 9, (2, 2))
    obj = (1, {'a': 3, 'b': 4, 'c': ('5', 6., (7, 8)), 'd': 9}, {}, arr, (),
           Dummy, Dummy())
    obj_orig = deepcopy(obj)

    copied = deepcopy_v2(obj, _obj_to_str)
    assert str(copied) == ("(1, {'a': 3, 'b': 4, 'c': ('5', 6.0, (7, 8)), "
                           "'d': 9}, {}, '[[7 5]\\n [1 8]]', (), "
                           "'Dummy', 'Dummy')")
    # deepcopy_v2 should not mutate original obj, but class instances will differ
    # by address
    assert str(obj[:-1]) == str(obj_orig[:-1])


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
