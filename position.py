'''A layer to add position encodings to a sequence.'''

from keras import backend as K
from keras.layers import Embedding

import numpy as np
import tensorflow as tf

def get_positions(x, mask_value=0.):
    '''Return a tensor of 1-indexed positions of each element in each sequence.

    x -- A padded batch of embedded sequences: example x position x dimension
         OR
         A padded batch of integers representing sequences: example x position
    '''
    mask = K.not_equal(x, mask_value)
    if len(x.shape) == 3:
        mask = K.any(mask, axis=-1)
    positions = K.cumsum(K.cast(mask, K.floatx()), axis=-1)
    return positions

def absolute_position_embeddings(x, max_length=None, mask_value=0.):
    '''Return absolute position embeddings.'''
    dimensions = x.get_shape().as_list()[-1]
    assert dimensions is not None, 'Last dimension must not be None'

    length = x.get_shape().as_list()[-2]
    if length is None:
        assert max_length is not None, 'Either max_length or length dimension must not be None'
        length = max_length

    positions = get_positions(x)
    position_embedding = Embedding(input_dim=length + 1, output_dim=dimensions)
    return position_embedding(positions)

def trig_position_embeddings(x, mask_value=0.):
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
