'''Minimal transformer problem.

Sorts digits.
'''

import itertools
import functools
import json
import os
import random
import shutil
import sys
from datetime import datetime

from argparse import ArgumentParser
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K

from position import *
from transformer import *

# HACK https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras
from keras import backend as K
K.set_learning_phase(1) #set learning phase

num_digits = 4
digit_len = 20
embed_dim = 256
num_layers = 1
delay = 7
data_len = 10000
epochs = 100
learning_rate = 0.1
momentum = 0.95

def gen_example(k):
    '''Generate an example sequence input-output pair of length k.'''
    input = [random.randint(1, num_digits) for _ in range(k)]
    x = [0] * delay + input
    output = [(x[i] + x[i+delay]) % num_digits + 1 for i, _ in enumerate(input)]
    return input, np_utils.to_categorical(output, num_digits+1)

def gen_dataset(n, k):
    '''Generate n examples.'''
    ins, outs = zip(*[gen_example(k) for _ in range(n)])
    return np.array(ins), np.array(outs)

x, y = gen_dataset(data_len, digit_len)

def create_model(embed_positions=None, get_encoder=None):
    """Return a model."""
    digits = Input(shape=(digit_len,), dtype='int32')
    digit_embedding = Embedding(input_dim=num_digits + 1, output_dim=embed_dim)
    embedded_digits = digit_embedding(digits)

    if embed_positions:
        embedded_positions = Lambda(embed_positions)(embedded_digits)
        embedded_digits = add([embedded_digits, embedded_positions])

    embedded_contexts = embedded_digits
    for i in range(num_layers):
        if get_encoder:
            embedded_contexts = get_encoder()(embedded_contexts)

    tagger = Dense(num_digits + 1, activation='softmax')
    tags = tagger(embedded_contexts)

    optimizer = optimizers.SGD(lr=learning_rate,
                               momentum=momentum,
                               decay=learning_rate/epochs)
    model = Model(inputs=digits, outputs=tags)
    model.compile(optimizer, categorical_crossentropy)
    return model

get_transformer = lambda: Transformer(embed_dim, heads=8, residual=True, dropout=0)
get_lstm = lambda: Bidirectional(LSTM(embed_dim // 2, return_sequences=True))
model = create_model(None, get_lstm)
model.fit(x, y, epochs=epochs)
