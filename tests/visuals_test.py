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

from backend import notify, _get_test_names
from deeptrain.visuals import (
    viz_roc_auc,
    )


tests_done = {}


@notify(tests_done)
def test_viz_roc_auc():
    y_true = np.array([1] * 5 + [0] * 27)  # imbalanced
    y_pred = np.random.uniform(0, 1, 32)
    viz_roc_auc(y_true, y_pred)


tests_done.update({name: None for name in _get_test_names(__name__)})

if __name__ == '__main__':
    pytest.main([__file__, "-s"])
