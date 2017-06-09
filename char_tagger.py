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
from tags import *
from subwords import *
from multi_model import *
from remove_mask import *

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
word_size = 150


def create_model_fn():
    """Return a function that builds a model for a max sentence length."""
    # Define all layers so that parameters persist across calls to inner fn
    char_embedding = Embedding(input_dim=num_chars,
                               output_dim=char_size,
                               mask_zero=True)
    word_embedding = Bidirectional(LSTM(lstm_size))
    context_embedding = Bidirectional(LSTM(lstm_size, return_sequences=True))
    encoder = Dense(word_size)
    tagger = Dense(num_tags, activation='softmax')
    sgd = optimizers.SGD(lr=0.2, momentum=0.95)

    @functools.lru_cache(None)
    def for_input(sentence_len):
        """Return a model and output layer for an input representing chars."""
        chars = Input(shape=(sentence_len, max_word_len), dtype='int32')
        embedded_chars = TimeDistributed(char_embedding)(chars)
        embedded_words = TimeDistributed(word_embedding)(embedded_chars)
        encoded_contexts = encoder(context_embedding(Masking()(embedded_words)))
        tags = RemoveMask()(tagger(encoded_contexts))

        model = Model(inputs=chars, outputs=tags)
        model.compile(optimizer=sgd,
                      loss=padded_categorical_crossentropy,
                      metrics=[padded_categorical_accuracy])
        return model

    return for_input

model_for_length = create_model_fn()


def model_for_x(x):
    """Return a model based on a function argument x."""
    if type(x) == list:
        return model_for_x(x[0])
    return model_for_length(x.shape[1])

model = MultiModel(model_for_length(250), model_for_x)

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

def prep(tagged_sents, max_len=None):
    """Convert a dataset of tagged sentences into inputs and outputs."""
    if not max_len:
        tagged_sents = list(tagged_sents)
        max_len = max(len(t) for t in tagged_sents)
    assert all(len(t) <= max_len for t in tagged_sents)
    x = np.array([prep_chars(t, max_len) for t in tagged_sents])
    tags = tag_map.texts_to_sequences(map(tag_string, tagged_sents))
    padded_tags = sequence.pad_sequences(tags, maxlen=max_len, value=0)
    y = np.array([np_utils.to_categorical(t, num_tags) for t in padded_tags])
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
            yield prep(contents, bin_len)
            groups.pop(bin_len)
    for bin_len, contents in groups.items():
        yield prep(contents, bin_len)

def list_tagged(corpus):
    return list(tagged_sents([corpus]))

xy_batches = list(grouped_batches(list_tagged(ptb_train)))
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

def shuffled_batch_generator(batches):
    """Yield batches in shuffled order repeatedly."""
    while True:
        random.shuffle(batches)
        yield from batches

early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
                               min_delta=0.0005, patience=0, verbose=1)
checkpoint = ModelCheckpoint(output_dir() + '/checkpoint.{epoch:02d}.hdf5')

def train(model):
    return model.fit_generator(shuffled_batch_generator(xy_batches),
                               steps_per_epoch=len(xy_batches),
                               epochs=30,
                               verbose=1,
                               validation_data=val,
                               callbacks=[checkpoint]).history

def evaluate(model, history):
    """Evaluate a model on all test sets."""
    losses = []
    accs = []
    for name, data in zip(['val', 'test'], [val, test]):
        loss = model.evaluate(*data, verbose=2)
        losses.append((name, loss))
        accs.append('{:0.4f}'.format(loss[1]))
        print('{}: loss: {:0.4f} - acc: {:0.4f}'.format(name, *loss))

    print('\t'.join(accs)) # For easy spreadsheet copy/paste

    with open(output_dir() + '/info.json', 'w') as jout:
        info = {
            'history': history,
            'losses': losses,
            'sys.argv': sys.argv,
        }
        json.dump(info, jout, indent=2)

    shutil.copyfile('tagger.py', output_dir() + '/tagger.py')

history = train(model)
evaluate(model, history)
