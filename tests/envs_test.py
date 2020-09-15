# -*- coding: utf-8 -*-
import os
import sys
# ensure `tests` directory path is on top of Python's module search
filedir = os.path.dirname(__file__)
sys.path.insert(0, filedir)
while filedir in sys.path[1:]:
    sys.path.pop(sys.path.index(filedir))  # avoid duplication

import pytest

from copy import deepcopy
from unittest.mock import patch
from importlib import reload

from backend import CL_CONFIGS, tempdir, notify, make_classifier
from backend import _init_session, _get_test_names
from backend import TF_EAGER, TF_KERAS
from deeptrain import introspection


#### HELPER METHODS ##########################################################
class ImportRaiser():
    def __init__(self, module_names):
        if not isinstance(module_names, (list, tuple)):
            module_names = [module_names]
        self.module_names = module_names

    def find_spec(self, fullname, path, target=None):
        if fullname in self.module_names:
            raise ImportError()


def _fail_import(fail_module, reload_module):
    module = sys.modules.pop(fail_module, None)
    sys.meta_path.insert(0, ImportRaiser(fail_module))

    reload(reload_module)

    if module is not None:
        sys.modules[fail_module] = module
    sys.meta_path.pop(0)

#### CONFIGURE TESTING #######################################################
tests_done = {}
CONFIGS = deepcopy(CL_CONFIGS)
batch_size, width, height, channels = CONFIGS['model']['batch_shape']

classifier = make_classifier(**CONFIGS['model'])

def init_session(C, weights_path=None, loadpath=None, model=None):
    return _init_session(C, weights_path=weights_path, loadpath=loadpath,
                         model=model, model_fn=make_classifier)
###############################################################################

@notify(tests_done)
@patch('deeptrain.introspection._make_grads_fn')
@patch('deeptrain.introspection.tf')
def test_tf_graph(MockClass1, MockClass2):
    """Call `_fn_graph` within `_make_grads_fn`."""
    if not (TF_KERAS and TF_EAGER):
        return

    MockClass1.executing_eagerly = lambda: False
    MockClass2.return_value = lambda *x: x
    C = deepcopy(CONFIGS)

    with tempdir(C['traingen']['logs_dir']), \
        tempdir(C['traingen']['best_models_dir']):
        tg = init_session(C, model=classifier)
        grads_fn = introspection._make_gradients_fn(tg.model, 0, 'outputs')
        grads_fn(0, 0, 0)


@patch.dict('sys.modules', dict(PIL=os))
def test_imports():
    tf_k = os.environ['TF_KERAS']
    os.environ['TF_KERAS'] = '0'

    from deeptrain.util import _backend
    reload(_backend)
    _backend.Unbuffered(sys.stdout).writelines("woot")

    _fail_import('lz4framed', _backend)
    assert _backend.lz4f is None

    os.environ['TF_KERAS'] = tf_k


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
