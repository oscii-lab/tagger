"""Sequence loss that ignores padding."""

import numpy as np
import tensorflow as tf

from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import backend as K

def padded_categorical_crossentropy(y_true, y_pred):
    """Cross-entropy loss of a batch, ignoring padding.

    >>> true = tf.constant([0., 0., 1., 1., 0., 0.], shape=[1, 2, 3])
    <1 example batch, 2 time steps, 3 tags: [2, 0 (padding)]>
    >>> pred = tf.constant([0, 0.5, 0.5, 0.4, 0.4, 0.2], shape=[1, 2, 3])
    >>> padded_categorical_crossentropy(true, pred) # => [[ 0.69 0.]]
    """
    full_loss = categorical_crossentropy(y_true, y_pred)
    padded = tf.squeeze(tf.slice(y_true, [0, 0, 0], [-1, -1, 1]), axis=2)
    mask = 1. - padded
    return full_loss * mask

def padded_categorical_accuracy(y_true, y_pred):
    """Accuracy of a batch, ignoring padding.

    >>> sh = [1, 3, 3] # 1 batch size x 3 time steps x 3 categories
    >>> true = tf.constant([0., 0., 1., 0., 0., 1., 1., 0., 0.], shape=sh)
    <[2, 2, 0 (padding)]>
    >>> pred = tf.constant([0, 0.7, 0.3, 0, 0.3, 0.7, 0.5, 0.3, 0.2], shape=sh)
    <[2, 1, 0)]>
    >>> padded_categorical_accuracy(true, pred).eval()) # => 0.5
    """
    padded = tf.squeeze(tf.slice(y_true, [0, 0, 0], [-1, -1, 1]), axis=2)
    mask = K.equal(padded, 0.)
    return categorical_accuracy(tf.boolean_mask(y_true, mask),
                                tf.boolean_mask(y_pred, mask))
