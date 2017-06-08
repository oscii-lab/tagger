# Part-of-speech tagger over characters

import itertools
import functools
import json
import os
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
from keras.callbacks import EarlyStopping, ProgbarLogger
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K


from loss import *
from tags import *
from subwords import *

# Sizes, including mask value
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

def create_model():
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
    def for_input(chars):
        """Return a model and output layer for an input representing chars."""
        embedded_chars = TimeDistributed(char_embedding)(chars)
        embedded_words = TimeDistributed(word_embedding)(embedded_chars)
        encoded_contexts = encoder(context_embedding(Masking()(embedded_words)))
        tags = tagger(encoded_contexts)

        model = Model(inputs=chars, outputs=tags)
        model.compile(optimizer=sgd,
                      loss=categorical_crossentropy,
                      metrics=[categorical_accuracy])
        return model, tags

    return for_input

@functools.lru_cache(None)
def input_for_len(sentence_len):
    return Input(shape=(sentence_len, max_word_len), dtype='int32')

def model_for_length(sentence_len, model_for_input=create_model()):
    """Return a model for a maximum sentence length."""
    return model_for_input(input_for_len(sentence_len))

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
    return x[:100], y[:100]

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

def grouped_batches(examples, prep, choose_bin):
    """Generate batches grouped by length."""
    groups = {}
    for example in examples:
        bin_len = choose_bin(example)
        bin_contents = groups.setdefault(bin_len, [])
        bin_contents.append(example)
        if (len(bin_contents) + 1) * bin_len > words_per_batch:
            yield prep(bin_contents, bin_len)
            groups.pop(bin_len)
    for bin_len, bin_contents in groups.items():
        yield prep(bin_contents, bin_len)

xy_batches = list(grouped_batches(tagged_sents([ptb_train]), prep, choose_bin))
val = prep(tagged_sents([ptb_dev]))
test = prep(tagged_sents([ptb_test]))

# %%
# Train and evaluate.

early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
                               min_delta=0.0005, patience=0, verbose=1)
def train(model):
    return model.fit_generator(itertools.cycle(xy_batches),
                               steps_per_epoch=len(xy_batches),
                               epochs=5,
                               verbose=1,
                               validation_data=val,
                               callbacks=[early_stopping]).history

def evaluate(label, model, history, exp_dir='exp'):
    """Evaluate a labeled model on all test sets and save it."""
    print('Evaluating', label)
    losses = []
    accs = []
    for name, data in zip(['val', 'test'], [val, test]):
        loss = model.evaluate(*data, verbose=2)
        losses.append((name, loss))
        accs.append('{:0.4f}'.format(loss[1]))
        print('{}: loss: {:0.4f} - acc: {:0.4f}'.format(name, *loss))

    print('\t'.join(accs)) # For easy spreadsheet copy/paste

    output_dir = datetime.today().strftime(exp_dir + '/%y%m%d_%H%M%S')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    os.mkdir(output_dir)

    try:
        with open(output_dir + '/model.json', 'w') as jout:
            jout.write(model.to_json())
        model.save(output_dir + '/model.h5')
    except:
        pass

    with open(output_dir + '/info.json', 'w') as jout:
        info = {
            'history': history,
            'losses': losses,
            'sys.argv': sys.argv,
            'label': label,
        }
        json.dump(info, jout, indent=2)

    shutil.copyfile('tagger.py', output_dir + '/tagger.py')

class MultiModel(Model):
    """A wrapper that chooses a model based on method input x.

    example_model -- The model to use when the input is unknown.
    choose_model  -- A fn from x (an input tensor) to a model.
    """
    def __init__(self, example_model, choose_model):
        self.example_model = example_model
        self.choose_model = choose_model

        # If an attribute is not found on the example, look in this object
        example_model.__getattr__ = self.__getattribute__

    def __getattribute__(self, name):
        """Return instance attribute, or x-specific attr, or example attr."""
        if name == '__dict__' or name in self.__dict__.keys():
            return object.__getattribute__(self, name)

        a = self.example_model.__getattribute__(name)
        if hasattr(a, '__func__'): # It's a method!
            varnames = a.__func__.__code__.co_varnames
            if 'x' in varnames:
                x_pos = varnames.index('x')

                @functools.wraps(a)
                def late_binding_method(*args, **vargs):
                    x = vargs['x'] if 'x' in vargs else args[x_pos]

                    try:
                        x_specific = self.choose_model(x)
                    except:
                        print("Cannot choose a model from", x, "for", name)
                        x_specific = self.example_model

                    # If an attribute is not found, look in this object
                    x_specific.__getattr__ = self.__getattribute__

                    return x_specific.__getattribute__(name)(*args, **vargs)

                return late_binding_method
            else:
                return object.__getattribute__(self, name)
        return a

def model_for_x(x):
    """Return a model based on a function argument x."""
    if type(x) == list:
        return model_for_x(x[0])
    return model_for_length(x.shape[1])[0]

model = MultiModel(model_for_length(250)[0], model_for_x)

train(model)
