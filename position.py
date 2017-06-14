'''A layer to add position encodings to a sequence.'''

from keras import backend as K
from keras.layers import Lambda
import numpy as np

def add_position_encodings(x, mask_value=0.):
    '''For each input vector in x, add a vector encoding its sequence position.

    Description: https://arxiv.org/abs/1706.03762

    In the paper, sin and cos are interleaved, but here they are concatenated.
    '''
    dimensions = x.get_shape().as_list()[-1]
    assert dimensions is not None, 'Last dimension must not be None'
    half = dimensions // 2

    # One constant vector per position
    pos = K.cumsum(K.ones_like(x), axis=1)[:, :, :half]

    # A range over dimensions, repeated once per position
    i = (K.cumsum(K.ones_like(x), axis=2)[:, :, :half] - 1) * 2

    trig_arg = pos / (10000 ** (i / dimensions))
    encoding = K.concatenate((K.sin(trig_arg), K.cos(trig_arg)), axis=-1)

    boolean_mask = K.any(K.not_equal(x, mask_value), axis=-1, keepdims=True)
    output = x + encoding * K.cast(boolean_mask, K.floatx())
    return output

AddPositionEncodings = Lambda(add_position_encodings)
