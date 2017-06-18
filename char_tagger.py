# Part-of-speech tagger over characters

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

from loss import *
from position import *
from subwords import *
from tags import *
from transformer import *

#%%
# Define vocab sizes, including pad value

num_words = len(word_map.word_index) + 1
num_chars = len(char_map.word_index) + 1
num_tags =  len(tag_map.word_index) + 1

print('Word set size:', num_words-1)
print('Char set size:', num_chars-1)
print('Tag set size: ', num_tags-1)

# %%
# Build models.

max_word_len = 20
lstm_size = 150
char_size = 50
word_size = 128
model_type = ['lstm', 'transformer'][1]
num_layers = 6

chars = Input(shape=(None, max_word_len), dtype='int32')


def LayerNormalize(inputs):
    g = tf.get_variable(
            name="g",
            shape=[inputs.get_shape()[2]],
            initializer=tf.constant_initializer(0.1))
    b = tf.get_variable(
            name="b",
            shape=[inputs.get_shape()[2]],
            initializer=tf.constant_initializer(0))

    mean = tf.reduce_mean(inputs, -1, keep_dims=True)

    deviation = inputs - mean
    x = tf.square(deviation)
    stdDev = tf.sqrt(tf.reduce_mean(x, axis = 2, keep_dims=True))

    return (g / stdDev) * deviation + b

def create_model():
    """Return a model."""
    char_embedding = Embedding(input_dim=num_chars,
                               output_dim=char_size,
                               mask_zero=True)
    embedded_chars = TimeDistributed(char_embedding)(chars)

    char_context = Bidirectional(LSTM(lstm_size))
    word_encoder = Dense(word_size)
    embedded_words = word_encoder(TimeDistributed(char_context)(embedded_chars))

    if model_type == 'lstm':
        word_context = Bidirectional(LSTM(lstm_size, return_sequences=True))
        context_encoder = Dense(word_size, activation='tanh')
        embedded_contexts = context_encoder(word_context(Masking()(embedded_words)))
    elif model_type == 'transformer':
        embedded_contexts = Dropout(.1)(AddPositionEncodings(embedded_words))
        for i in range(num_layers):
            embedded_contexts = Transformer(word_size, residual=True)(embedded_contexts)
            # embedded_contexts = BatchNormalization()(embedded_contexts)
            embedded_contexts = Lambda(LayerNormalize)(embedded_contexts)

    # embedded_words_and_position = Dropout(.1)(AddPositionEncodings(embedded_words))
    embedded_contexts = merge([embedded_contexts,embedded_words],mode='sum')

    tagger = Dense(num_tags, activation='softmax')
    tags = tagger(embedded_contexts)

    #optimizer = optimizers.SGD(lr=0.2, momentum=0.95)
    optimizer = optimizers.Adam()
    model = Model(inputs=chars, outputs=tags)
    model.compile(optimizer, categorical_crossentropy)
    return model

model = create_model()

# %%
# Prepare data format for model.

def shorten_word(w, max_word_len):
    """Shorten very long words."""
    slash = '\\/'
    if len(w) > max_word_len and slash in w:
        # slashed words represented by their first element
        w = w.split(slash)[0]
    if len(w) > max_word_len and '-' in w:
        # hyphenatied words represented by their last element
        parts = w.split('-')
        w = '_-' + parts[-1]
    if len(w) > max_word_len:
        # very long words represented by their prefix and suffix
        half = max_word_len//2
        w = w[:half] + '_' + w[-half+1:]
    return w

def word_string(tagged):
    return str(' '.join(w for w, t in tagged))

def tag_string(tagged):
    return str(' '.join(t for w, t in tagged))

def char_strings(tagged):
    return [' '.join(shorten_word(w, max_word_len)) for w, t in tagged]

def prep(tagged_sents):
    """Convert a dataset of tagged sentences into inputs and outputs."""
    tagged_sents = list(tagged_sents)
    max_len = max(len(t) for t in tagged_sents)
    x = np.array([prep_chars(t, max_len) for t in tagged_sents])
    tags = tag_map.texts_to_sequences(map(tag_string, tagged_sents))
    padded_tags = sequence.pad_sequences(tags, maxlen=max_len, value=0)
    y = np.array([np_utils.to_categorical(t, num_tags) for t in padded_tags]).astype(np.float32)
    return x, y

def prep_chars(tagged, max_len):
    """Convert sentence into a padded array of character embeddings."""
    chars = char_map.texts_to_sequences(char_strings(tagged))
    padded_chars = sequence.pad_sequences(chars, maxlen=max_word_len, value=0)
    padding = np.zeros([max_len-len(chars), max_word_len])
    return np.append(padding, padded_chars, axis=0).astype(np.int32)

bins = list(range(10, 61, 5)) + [90, max(len(t) for t in all_tagged)]
words_per_batch = 3000

@functools.lru_cache(None)
def next_largest(length):
    return next(b for b in bins if b >= length)

def choose_bin(tagged):
    return next_largest(len(tagged))

def grouped_batches(examples):
    """Generate batches grouped by length."""
    # TODO fn could be paramaterized by choose_bin, prep, and the yield condition.
    groups = {}
    for example in examples:
        bin_len = choose_bin(example)
        contents = groups.setdefault(bin_len, [])
        contents.append(example)
        if (len(contents) + 1) * bin_len > words_per_batch or len(contents) >= 100:
            yield prep(contents)
            groups.pop(bin_len)
    for bin_len, contents in groups.items():
        yield prep(contents)

def list_tagged(corpus):
    return list(tagged_sents([corpus]))

train_list = list_tagged(ptb_train)
train_plain = prep(list_tagged(ptb_train))

val = prep(list_tagged(ptb_dev))
test = prep(list_tagged(ptb_test))

# %%
# Train and evaluate.

@functools.lru_cache(None)
def output_dir(exp_dir='exp'):
    d = datetime.today().strftime(exp_dir + '/%y%m%d_%H%M%S')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    os.mkdir(d)
    return d

def shuffled_batch_generator(examples):
    """Yield batches in shuffled order repeatedly."""
    while True:
        random.shuffle(examples)
        yield from grouped_batches(examples)

checkpoint_pattern = output_dir() + '/checkpoint.{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_pattern)

def accuracy_ratio(y_true, y_pred):
    """Categorical accuracy of padded values."""
    c_true, c_pred = np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)
    n = sum(np.logical_and(c_true > 0, c_true == c_pred).reshape((-1,)))
    d = sum((c_true > 0).reshape((-1,)))
    return n, d

def compute_accuracy(model, name, epoch, result, log, x, y_true):
    """Compute and print accuracy."""
    y_pred = model.predict(x, batch_size=100, verbose=1)
    n, d = accuracy_ratio(y_true, y_pred)
    acc = np.round(100*n/d, 5)
    msg = '{} epoch {} accuracy: {}/{} ({}%)'.format(name, epoch, n, d, acc)
    print(msg)
    print(msg, file=log)
    result.append(n/d)
    return n/d

def train(model):
    batches = shuffled_batch_generator(train_list)
    num_batches = len(list(grouped_batches(train_list)))
    val_accs = []
    test_accs = []

    for k in range(1, 20):
        model.fit_generator(batches,
                            steps_per_epoch=num_batches,
                            epochs=k,
                            initial_epoch=k-1,
                            callbacks=[])
        with open(output_dir() + '/log.txt', 'a') as log:
            acc = compute_accuracy(model, 'train', k, val_accs, log, train_plain[0][:1000],train_plain[1][:1000])
        if k > 4:
            with open(output_dir() + '/log.txt', 'a') as log:
                acc = compute_accuracy(model, 'val', k, val_accs, log, *val)
                if acc > .96:
                    compute_accuracy(model, 'test', k, test_accs, log, *test)

    best_iter = np.argmax(val_accs)
    print('Best val:', val_accs[best_iter], 'test:', test_accs[best_iter])
    with open(output_dir() + '/log.txt', 'a') as log:
        print('Best val:', val_accs[best_iter], 'test:', test_accs[best_iter], file=log)

if __name__ == '__main__':
    train(model)
