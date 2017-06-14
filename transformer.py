'''A transformer layer.'''

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Transformer(Layer):
    def __init__(self, qkv_dim=64, heads=8, output_dim=1024, **kwargs):
        self.qkv_dim = qkv_dim
        self.heads = heads
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) == 3:
                qdim, kdim, vdim = [s[-1] for s in input_shape]
            elif len(input_shape) == 2:
                qdim, kdim, vdim = [s[-1] for s in [input_shape[0], input_shape[1], input_shape[1]]]
            elif len(input_shape) == 1:
                qdim, kdim, vdim = [s[-1] for s in [input_shape[0]] * 3]
        else:
            qdim, kdim, vdim = [input_shape[-1]] * 3

        def add(n, s):
            return self.add_weight(name=n, shape=s, initializer='uniform', trainable=True)

        self.projections = []
        for i in range(self.heads):
            self.projections.append([
                add('qp_%d' % i, (qdim, self.qkv_dim)),
                add('kp_%d' % i, (kdim, self.qkv_dim)),
                add('vp_%d' % i, (vdim, self.qkv_dim))
            ])
        encoding_dim = self.qkv_dim * self.heads
        self.inner_ffa = add('inner_ffa_%d' % i, (encoding_dim, self.output_dim))
        self.inner_ffb = add('inner_ffb_%d' % i, (self.output_dim,))
        self.outer_ffa = add('outer_ffa_%d' % i, (self.output_dim, self.output_dim))
        self.outer_ffb = add('outer_ffb_%d' % i, (self.output_dim,))
        super().build(input_shape)

    def call(self, x):
        if isinstance(x, list):
            if len(x) == 3:
                q, k, v = x
            elif len(x) == 2:
                q, k, v = x[0], x[1], x[1]
            else:
                q, k, v = x[0], x[0], x[0]
        else:
            q, k, v = x, x, x

        encodings = []
        for qp, kp, vp in self.projections:
            queries = K.dot(q, qp)
            keys = K.dot(k, kp)
            values = K.dot(v, vp)
            logits = K.batch_dot(queries, K.permute_dimensions(keys, [0, 2, 1]))
            distributions = K.softmax(logits / K.constant(np.sqrt(self.qkv_dim)))
            weighted_values = K.batch_dot(distributions, values)
            encodings.append(weighted_values)
        encoding = K.concatenate(encodings)
        # import pdb; pdb.set_trace()
        linear = K.dot(encoding, self.inner_ffa)
        K.bias_add(linear, self.inner_ffb)
        relu = K.maximum(0., linear)
        output = K.dot(relu, self.outer_ffa)
        K.bias_add(output, self.outer_ffb)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)
