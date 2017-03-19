# Part-of-speech tagger

import itertools
import functools
import json
import os
import shutil
import sys
from datetime import datetime

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
from keras import backend as K

from loss import *
from tags import *
from subwords import *

print('Vocab size:', len(word_map.word_counts))
print('Tag set size:', len(tag_map.word_counts))

# %%
# Load subwords

size = '8k'
subword_paths = [
    'data/ptb_all.' + size + '.txt',
    'data/web_all.' + size + '.txt',
    ]

if len(sys.argv) > 1 and not sys.argv[1] == '-f':
    subword_paths = sys.argv[1:]

# Subword representations; each takes a list of strings: subwords for a word.

def last_delimiter(subwords):
    if len(subwords) > 1:
        return subwords[:-1] + ['@@' + subwords[-1]]
    else:
        return subwords

def no_delimiter(subwords):
    return [s.strip('@@') for s in subwords]

def length(subwords):
    length = '@@{}@@'.format(len(subwords))
    return [length] + [s.strip('@@') for s in subwords[:max_subwords-1]]

def join(subwords):
    return [''.join([s.strip('@@') for s in subwords])]

# Define two different representations of subwords using the functions above.

max_subwords = 10
print('subword_paths:', subword_paths)
sub_with = lambda f: map_and_subworder(texts, subword_paths, max_subwords, f)
subword_map1, subworder1 = sub_with(last_delimiter)
subword_map2, subworder2 = sub_with(no_delimiter)

print(subworder1[texts[0]])
print(subworder2[texts[0]])

# %%
# Build models.

num_words = len(word_map.word_index) + 1
num_subwords1 = len(subword_map1.word_index) + 1
num_subwords2 = len(subword_map2.word_index) + 1
num_tags = len(tag_map.word_index) + 1
max_len = max(w.count(' ') + 1 for w in texts)
# max_subwords = max(w.count(' ') + 1 for s in subworder.values() for w in s)
word_size = 64

subwords1 = Input(shape=(max_len, max_subwords), dtype='int32')
subwords2 = Input(shape=(max_len, max_subwords), dtype='int32')

def make(drop1, drop2):
    # Embed each subword
    embed1 = Embedding(input_dim=num_subwords1, output_dim=word_size, mask_zero=True, dropout=drop1)
    embed2 = Embedding(input_dim=num_subwords2, output_dim=word_size, mask_zero=True, dropout=drop2)
    embedded1 = TimeDistributed(embed1)(subwords1)
    embedded2 = TimeDistributed(embed2)(subwords2)

    # Concatenate both embeddings & embed each subword sequence into a word vector.
    embedded12 = merge([embedded1, embedded2], mode='sum')
    embedded = TimeDistributed(LSTM(word_size))(embedded12)

    # Build a convolutional network
    convolved_words = Convolution1D(word_size, 5, border_mode='same')(embedded)
    merged_words = merge([embedded, convolved_words], mode='sum')

    # Predict tags from words
    tagger = Dense(num_tags, activation='softmax')
    tagged = tagger(merged_words)

    # Predict tags again from words and tags
    convolved_tags = Convolution1D(word_size, 5, border_mode='same')(tagged)
    merged = merge([merged_words, convolved_tags], mode='sum')
    tags = tagger(merged)

    model = Model(input=[subwords1, subwords2], output=tags)
    model.compile(optimizer='adam',
                  loss=padded_categorical_crossentropy,
                  metrics=[padded_categorical_accuracy])
    return model, tags

# Note: Put just one (label, model) pair in models for a simple experiment.
models = []
drops = [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.2, 0.1]]
for drop in drops:
    models.append((str(drop), make(*drop)))

# %%
# Prepare data format for model.

def word_string(tagged):
    return str(' '.join(w for w, t in tagged))

def tag_string(tagged):
    return str(' '.join(t for w, t in tagged))

def prep(tagged_sents):
    """Convert a dataset of tagged sentences into inputs and outputs."""
    tagged_sents = list(tagged_sents) # because we'll iterate twice
    one, two = [(subword_map1, subworder1), (subword_map2, subworder2)]
    x1 = np.array([prep_subword(word_string(t), *one) for t in tagged_sents])
    x2 = np.array([prep_subword(word_string(t), *two) for t in tagged_sents])
    x = [x1, x2]
    tags = tag_map.texts_to_sequences(map(tag_string, tagged_sents))
    padded_tags = sequence.pad_sequences(tags, maxlen=max_len, value=0)
    y = np.array([np_utils.to_categorical(t, num_tags) for t in padded_tags])
    return x, y

def prep_subword(sentence, subword_map, subworder):
    """Convert sentence into a padded array of subword embeddings."""
    subs = subword_map.texts_to_sequences(subworder[sentence])
    padded_subs = sequence.pad_sequences(subs, maxlen=max_subwords, value=0)
    padding = np.zeros([max_len-len(subs), max_subwords])
    return np.append(padding, padded_subs, axis=0).astype(np.int32)

x, y = prep(tagged_sents([ptb_train]))
val = prep(tagged_sents([ptb_dev]))
test = prep(tagged_sents([ptb_test]))
web_tests = [prep(tagged_sents([w])) for w in web_all]

# %%
# Train and evaluate.

early_stopping = EarlyStopping(monitor='val_padded_categorical_accuracy',
                               min_delta=0.0005, patience=0, verbose=1)
def train(model):
    return model.fit(x, y, batch_size=32, nb_epoch=4, verbose=1,
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
    model = Model(input=[subwords1, subwords2], output=ensemble)
    model.compile(optimizer='adam',
                  loss=padded_categorical_crossentropy,
                  metrics=[padded_categorical_accuracy])
    evaluate('ensemble', model, None)
