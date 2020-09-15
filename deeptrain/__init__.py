import os

def _get_scales():
    s = os.environ.get('SCALEFIG', '1')
    os.environ['SCALEFIG'] = s
    if ',' in s:
        w_scale, h_scale = map(float, s.strip('[()]').split(','))
    else:
        w_scale, h_scale = float(s), float(s)
    return w_scale, h_scale

def scalefig(fig):
    """Used internally to scale figures according to env var 'SCALEFIG'.

    os.environ['SCALEFIG'] can be an int, float, tuple, list, or bracketless
    tuple, but must be a string: '1', '1.1', '(1, 1.1)', '1,1.1'.
    """
    w, h = fig.get_size_inches()
    w_scale, h_scale = _get_scales()  # refresh in case env var changed
    fig.set_size_inches(w * w_scale, h * h_scale)

# get at base package level (deeptrain) to set for see_rnn
_get_scales()

##############################################################################
from . import train_generator
from . import data_generator
from . import metrics
from . import callbacks
from . import visuals
from . import introspection
from . import preprocessing
from . import util

from .train_generator import TrainGenerator
from .data_generator import DataGenerator


def set_seeds(seeds=None, reset_graph=False, verbose=1):
    """Sets random seeds and maybe clears keras session and resets TensorFlow
    default graph.

    NOTE: after calling w/ `reset_graph=True`, best practice is to re-instantiate
    model, else some internal operations may fail due to elements from different
    graphs interacting (of pre-reset model and post-reset tensors).
    """
    callbacks.RandomSeedSetter._set_seeds(seeds, reset_graph, verbose)


__version__ = '0.6.0'
__author__ = 'OverLordGoldDragon'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2020, %s.' % __author__
__homepage__ = 'https://github.com/OverLordGoldDragon/deeptrain'
__docs__ = (
    "DeepTrain: Full knowledge and control of the train state."
)
