'''A transformer layer.'''

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers.core import Dense, Dropout
from keras.activations import softmax

import numpy as np
import tensorflow as tf


class Transformer(Dense):
    '''Transformer layer.

    Description: https://arxiv.org/abs/1706.03762

    Implemented as a sub-class of Dense for initializer args, etc. The base
    class weights apply the final linear projection in the transformer.
    '''
    def __init__(self, units, hidden_units=None, heads=8, residual=True, mask_value=0, dropout=0, **kwargs):
        assert units, 'Units must be positive'
        assert units % heads == 0, 'Number of heads must evenly divide units'

        if hidden_units is None:
            self.hidden_units = units
        else:
            self.hidden_units = hidden_units

        self.heads = heads
        self.projected_dim = units // heads
        self.residual = residual
        self.mask_value = mask_value
        self.dropout = Dropout(dropout)
        kwargs['use_bias'] = True
        super().__init__(units, **kwargs)

    @staticmethod
    def _expand_qkv(value_or_list):
        '''Expand one or more arguments into a three-element list.'''
        if isinstance(value_or_list, list):
            if len(value_or_list) == 3:    # query, key, value
                return value_or_list
            elif len(value_or_list) == 2:  # query, key_and_value
                return [value_or_list[0], value_or_list[1], value_or_list[1]]
            elif len(value_or_list) == 1:
                return [value_or_list[0]] * 3
        else:
            return [value_or_list] * 3

    def _add(self, name, shape):
        '''Add a weight.'''
        return self.add_weight(name=name,
                               shape=shape,
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)

    def _normalize(self, inputs, suffix=''):
        mean = tf.reduce_mean(inputs, -1, keep_dims=True)
        deviation = inputs - mean
        x = tf.square(deviation)
        stdDev = tf.sqrt(tf.reduce_mean(x, axis=2, keep_dims=True))
        g, b = [getattr(self, var + suffix) for var in ['g', 'b']]
        return (g / stdDev) * deviation + b

    def build(self, input_shape):
        shapes = self._expand_qkv(input_shape)
        qdim, kdim, vdim = [s[-1] for s in shapes]
        self.projections = []
        for i in range(self.heads):
            self.projections.append([
                self._add('qp_%d' % i, (qdim, self.projected_dim)),
                self._add('kp_%d' % i, (kdim, self.projected_dim)),
                self._add('vp_%d' % i, (vdim, self.projected_dim)),
            ])

        self.g1 = self._add('g1', (self.units,))
        self.b1 = self._add('b1', (self.units,))
        self.g2 = self._add('g2', (self.units,))
        self.b2 = self._add('b2', (self.units,))
        self.relu_kernel = self._add('relu_kernel', (self.units, self.hidden_units))
        self.relu_bias = self._add('relu_bias', (self.hidden_units,))
        super().build(list(shapes[0])[:-1] + [self.units]) # Dense after relu

        if self.residual:
            # Ensure that input and output shapes match
            self.input_spec = InputSpec(min_ndim=2, axes={-1: self.units})
        else:
            self.input_spec = InputSpec(min_ndim=2)

    def call(self, x):
        q, k, v = self._expand_qkv(x)
        encodings = []
        for qp, kp, vp in self.projections:
            queries = K.dot(q, qp)
            keys = K.dot(k, kp)
            values = K.dot(v, vp)
            logits = K.batch_dot(queries, K.permute_dimensions(keys, [0, 2, 1]))
            distributions = softmax(logits / np.sqrt(self.projected_dim), axis=-1)

            # Make it impossible to attend to masked values.
            key_mask = K.any(K.not_equal(k, self.mask_value), axis=-1, keepdims=True)
            distributions *= K.cast(key_mask, K.floatx())
            distributions /= K.expand_dims(K.sum(distributions, axis=-1), axis=-1)

            distributions = self.dropout(distributions)

            encoding = K.batch_dot(distributions, values)
            encodings.append(encoding)
        encoding = K.concatenate(encodings)
        if self.residual:
            encoding = x + self.dropout(encoding)
        encoding = self._normalize(encoding, '1')

        linear = K.dot(encoding, self.relu_kernel)
        linear = K.bias_add(linear, self.relu_bias)
        relu = K.relu(linear)
        output = super().call(relu) # Applies dense layer
        if self.residual:
            output = encoding + output
        output = self._normalize(output, '2')

        return output

    def get_config(self):
        config = {
            'heads': self.heads,
            'residual': self.residual,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
