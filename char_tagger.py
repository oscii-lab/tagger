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
from keras.callbacks import EarlyStopping
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K


from loss import *
from tags import *
from subwords import *

print('Vocab size:', len(word_map.word_counts))
print('Tag set size:', len(tag_map.word_counts))

# %%
# Characters

words = [' '.join(w) for w in word_map.word_index.keys()]
char_map = Tokenizer(lower=False, filters='')
char_map.fit_on_texts(words)

check_for_oov_characters = False
if check_for_oov_characters:
    train_chars = set([c for w, t in ptb_train.tagged_words() for c in w])
    dev_chars = set([c for w, t in ptb_dev.tagged_words() for c in w])
    test_chars = set([c for w, t in ptb_test.tagged_words() for c in w])
    print(dev_chars - train_chars, test_chars-train_chars)

# %%
# Build models.

num_words = len(word_map.word_index) + 1
num_tags = len(tag_map.word_index) + 1
num_chars = len(char_map.word_index) + 1
max_len = max(len(t) for t in tagged_sents([ptb_dev, ptb_test]))
max_word_len = 20
lstm_size = 150
char_size = 50
word_size = 150

chars = Input(shape=(max_len, max_word_len), dtype='int32')

def encode(inputs, key='lstm'):
    if key == 'lstm':
        masked_inputs = Masking()(inputs)
        encoder = Bidirectional(LSTM(lstm_size, return_sequences=True))
        encoded = encoder(masked_inputs)
        return Dense(word_size)(encoded)
    raise ValueError

def make():
    # Embed each character sequence into a word vector.
    char_embedding = Embedding(input_dim=num_chars,
                               output_dim=char_size,
                               mask_zero=True)
    embedded_chars = TimeDistributed(char_embedding)(chars)
    word_embedding = Bidirectional(LSTM(lstm_size))
    embedded_words = TimeDistributed(word_embedding)(embedded_chars)

    # Encode words and predict tags
    encoded_words = encode(embedded_words, 'lstm')
    tagger = Dense(num_tags, activation='softmax')
    tags = tagger(encoded_words)

    sgd = optimizers.SGD(lr=0.2, momentum=0.95)
    model = Model(inputs=chars, outputs=tags)
    model.compile(optimizer=sgd,
                  loss=categorical_crossentropy,
                  metrics=[categorical_accuracy])

    return model, tags

models = [[str(k), make()] for k in range(8)]

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
    all_sents = list(tagged_sents)
    tagged_sents = [t for t in all_sents if len(t) <= max_len]
    discarded = len(all_sents) - len(tagged_sents)
    if discarded:
        print('Discarding', discarded, 'examples')
    x = np.array([prep_chars(t) for t in tagged_sents])
    tags = tag_map.texts_to_sequences(map(tag_string, tagged_sents))
    padded_tags = sequence.pad_sequences(tags, maxlen=max_len, value=0)
    y = np.array([np_utils.to_categorical(t, num_tags) for t in padded_tags])
    return x, y

def prep_chars(tagged):
    """Convert sentence into a padded array of character embeddings."""
    chars = char_map.texts_to_sequences(char_strings(tagged))
    padded_chars = sequence.pad_sequences(chars, maxlen=max_word_len, value=0)
    padding = np.zeros([max_len-len(chars), max_word_len])
    return np.append(padding, padded_chars, axis=0).astype(np.int32)

print('Preparing data; max_len =', max_len)
x, y = prep(tagged_sents([ptb_train]))
val = prep(tagged_sents([ptb_dev]))
test = prep(tagged_sents([ptb_test]))
print('Data shape for x:', x.shape, 'y:', y.shape)
# web_tests = [prep(tagged_sents([w])) for w in web_all]

# %%
# Train and evaluate.

early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
                               min_delta=0.0005, patience=0, verbose=1)
def train(model):
    return model.fit(x, y, batch_size=100, epochs=10, verbose=1,
                     validation_data=val, callbacks=[early_stopping]).history

def evaluate(label, model, history, exp_dir='exp'):
    """Evaluate a labeled model on all test sets and save it."""
    print('Evaluating', label)
    losses = []
    accs = []
    for name, data in zip(['val', 'test'] + web_genres, [val, test] + web_tests):
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

for label, (model, _) in models:
    history = train(model)
    evaluate(label, model, history)

if len(models) > 1:
    ensemble = merge([tags for (_, (_, tags)) in models], mode='ave')
    model = Model(input=subwords, output=ensemble)
    model.compile(optimizer='adam',
                  loss=categorical_crossentropy,
                  metrics=[categorical_accuracy])
    evaluate('ensemble', model, None)
