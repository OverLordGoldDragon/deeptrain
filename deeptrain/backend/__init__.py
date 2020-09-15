import tensorflow as tf
from termcolor import colored
from deeptrain.util._backend import TF_KERAS

WARN = colored('WARNING:', 'red')
NOTE = colored('NOTE:', 'blue')

#### Env flags & Keras backend ###############################################
tf_eager = tf.executing_eagerly
TF_2 = bool(tf.__version__[0] == '2')

if TF_KERAS:
    import tensorflow.keras.backend as K
else:
    import keras.backend as K

#### Subpackage imports ######################################################
from . import model_utils

from .model_utils import *



