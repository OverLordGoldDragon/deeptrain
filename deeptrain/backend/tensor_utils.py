from . import K, TF_KERAS, TF_2
from ..util.misc import try_except


def eval_tensor(x, backend):
    K = backend
    te = try_except
    return te(lambda x: K.get_value(K.to_dense(x)),
              te(lambda x: K.function([], [x])([])[0],
                 te(K.eager(K.eval),
                    K.eval)))
