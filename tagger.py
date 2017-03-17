# Convolutional tagger over subwords

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

size = '8k'
max_subwords = 10
subword_paths = [
    'data/ptb_all.' + size + '.txt',
    'data/web_all.' + size + '.txt',
    ]

if len(sys.argv) > 1:
    subword_paths = sys.argv[1:]

subword_map, subworder = map_and_subworder(texts, subword_paths, max_subwords)

# %%
# Build a model.

num_words = len(word_map.word_index)+1
num_subwords = len(subword_map.word_index)+1
num_tags = len(tag_map.word_index)+1
max_len = max(w.count(' ') + 1 for w in subworder.keys())
# max_subwords = max(w.count(' ') + 1 for s in subworder.values() for w in s)
word_size = 64

# Embed each subword
subwords = Input(shape=(max_len, max_subwords), dtype='int32', name='Subwords')
sub_embedding = Embedding(input_dim=num_subwords, output_dim=word_size, mask_zero=True)

# Embed subword sequences each into a single word vector.
embedded = TimeDistributed(LSTM(word_size))(TimeDistributed(sub_embedding)(subwords))

# Build a convolutional network
convolved = Convolution1D(word_size, 5, border_mode='same')(embedded)
representations = [embedded, convolved]
merged = merge(representations, mode='sum')
tags = Dense(num_tags, activation='softmax', name='Tag')(merged)

model = Model(input=subwords, output=tags)
model.compile(optimizer='adam',
              loss=padded_categorical_crossentropy,
              metrics=[padded_categorical_accuracy])

# %%
# Prepare data format for model.

def word_string(tagged):
    return str(' '.join(w for w, t in tagged))

def tag_string(tagged):
    return str(' '.join(t for w, t in tagged))

def prep(tagged_sents):
    """Convert a dataset of tagged sentences into inputs and outputs."""
    tagged_sents = list(tagged_sents) # because we'll iterate twice
    x = np.array([prep_subword(word_string(t)) for t in tagged_sents])
    tags = tag_map.texts_to_sequences(map(tag_string, tagged_sents))
    padded_tags = sequence.pad_sequences(tags, maxlen=max_len, value=0)
    y = np.array([np_utils.to_categorical(t, num_tags) for t in padded_tags])
    return x, y

def prep_subword(sentence):
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

early_stopping = EarlyStopping(monitor='val_padded_categorical_accuracy',
                               min_delta=0.001, patience=2, verbose=1)
history = model.fit(x, y, batch_size=32, nb_epoch=100, verbose=1,
                    validation_data=val, callbacks=[early_stopping])

losses = []
accs = []
for name, data in zip(['val', 'test'] + web_genres, [val, test] + web_tests):
    loss = model.evaluate(*data, verbose=2)
    losses.append((name, loss))
    accs.append('{:0.4f}'.format(loss[1]))
    print('{}: loss: {:0.4f} - acc: {:0.4f}'.format(name, *loss))

print('\t'.join(accs))


# %%
# Save everything

output_dir = datetime.today().strftime('exp/%y%m%d_%H%M%S')

if not os.path.exists('exp'):
    os.mkdir('exp')
os.mkdir(output_dir)

with open(output_dir + '/model.json', 'w') as jout:
    jout.write(model.to_json())

model.save(output_dir + '/model.h5')

with open(output_dir + '/info.json', 'w') as jout:
    info = {
        'history': history.history,
        'losses': losses,
        'sys.argv': sys.argv
    }
    json.dump(info, jout, indent=2)

shutil.copyfile('tagger.py', output_dir + '/tagger.py')
