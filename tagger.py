# Part-of-speech tagger

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
# Load subwords

size = '8k'
subword_paths = [
    'data/ptb_all.' + size + '.txt',
    'data/web_all.' + size + '.txt',
    ]



parser = ArgumentParser()
parser.add_argument("-f", "--filename",dest='subword_paths', help="Path for subwords")
parser.add_argument("-e", "--encoder",choices=["conv","lstm"],help="Choose convolutional or BiLSTM encoder")


args = parser.parse_args()


if args.subword_paths is not None:
    subword_paths = options.subword_paths



max_subwords = 25
print('subword_paths:', subword_paths)
subword_map, subworder = map_and_subworder(texts, subword_paths, max_subwords)

print(subworder[texts[0]])

# %%
# Build models.

num_words = len(word_map.word_index) + 1
num_subwords = len(subword_map.word_index) + 1
num_tags = len(tag_map.word_index) + 1
max_len = max(w.count(' ') + 1 for w in texts)
# max_subwords = max(w.count(' ') + 1 for s in subworder.values() for w in s)
lstm_size = 150
word_dim = 50
char_size = 50



subwords = Input(shape=(max_len, max_subwords), dtype='int32')

def encoder(x):
    if args.encoder == "conv":
        encoded = Convolution1D(lstm_size, 5, border_mode='same')(x)
        print('Encoder not supported')
        raise ValueError
    elif args.encoder == "lstm":
        encoded = Bidirectional(LSTM(lstm_size,return_sequences=True))(x)
    else:
        print('Encoder not supported')
        raise ValueError
    return encoded




def make(dropout=0, k=1, tag_twice=False):
    # Embed each subword k times
    embedded_subwords_list = []
    for _ in range(k):
        e = Embedding(input_dim=num_subwords, output_dim=char_size, mask_zero=True, dropout=dropout)
        embedded_subwords_list.append(TimeDistributed(e)(subwords))
    if k == 1:
        embedded_subwords = embedded_subwords_list[0]
    else:
        embedded_subwords = merge(embedded_subwords_list, mode='sum')

    # Embed each subword sequence into a word vector.
    bi_embedded_words = TimeDistributed(Bidirectional(LSTM(lstm_size)))(embedded_subwords)
    embedded_words = TimeDistributed(Dense(word_dim))(bi_embedded_words)

    # Build a convolutional network
    bi_encoded_words = encoder(embedded_words)
    encoded_words = TimeDistributed(Dense(lstm_size,activation='tanh'))(bi_encoded_words)

    # Predict tags from words
    tagger = Dense(num_tags, activation='softmax')
    tags = tagger(encoded_words)

    sgd = optimizers.SGD(lr=0.2, momentum=0.95)
    model = Model(input=subwords, output=tags)

    model.compile(optimizer=sgd,
                  loss=padded_categorical_crossentropy,
                  metrics=[padded_categorical_accuracy])
    return model, tags

# Note: Put just one (label, model) pair in models for a simple experiment.
repetitions = range(8)
models = [(str([0.0, k, 'tag_once']), make(0.0, 1, False)) for k in repetitions]
# models += [(str([0.2, k, 'tag_once']), make(0.2, 2, False)) for k in repetitions]
# models += [(str([0.0, k, 'tag_twice']), make(0.0, 1, True)) for k in repetitions]
# models += [(str([0.2, k, 'tag_twice']), make(0.2, 2, True)) for k in repetitions]
# models = [(str([0.2, k, 'tag_twice']), make(0.2, 2, True)) for k in repetitions]

# %%
# Prepare data format for model.

def word_string(tagged):
    return str(' '.join(w for w, t in tagged))

def tag_string(tagged):
    return str(' '.join(t for w, t in tagged))

def prep(tagged_sents):
    """Convert a dataset of tagged sentences into inputs and outputs."""
    tagged_sents = list(tagged_sents) # because we'll iterate twice
    subs = (subword_map, subworder)
    x = np.array([prep_subword(word_string(t), *subs) for t in tagged_sents])
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
    return model.fit(x, y, batch_size=100, nb_epoch=32, verbose=1,
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
                  loss=padded_categorical_crossentropy,
                  metrics=[padded_categorical_accuracy])
    evaluate('ensemble', model, None)
