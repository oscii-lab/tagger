# Convolutional tagger over subwords

import itertools
import functools

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

from nltk.corpus import BracketParseCorpusReader

from loss import *

# %%
# Load data.

# Create a data directory and then unarchive the following:
# - https://catalog.ldc.upenn.edu/LDC2015T13 (Penn Treebank Revised)
# - https://catalog.ldc.upenn.edu/LDC2012T13 (English Web Treebank)

# Standard part-of-speech tagging train/dev/test split
ptb_root = 'data/eng_news_txt_tbnk-ptb_revised/data/penntree'
ptb_train = BracketParseCorpusReader(ptb_root, r'(0[0-9]|1[0-8])/.*tree')
ptb_dev = BracketParseCorpusReader(ptb_root, r'(19|20|21)/wsj_.*tree')
ptb_test = BracketParseCorpusReader(ptb_root, r'(22|23|24)/wsj_.*tree')
ptb_all = BracketParseCorpusReader(ptb_root, r'[0-9]*/wsj_.*tree')

# %%
# Index all types as integers.

def tagged_sents(corpus):
    """Iterate over tagged sentences, dropping fake words such as *PRO*."""
    for ts in corpus.tagged_sents():
        yield [(w, t) for w, t in ts if not (w.startswith('*') and w.endswith('*'))]

ptb_all_tagged = list(tagged_sents(ptb_all))
texts = [str(' '.join(w for w, _ in s)) for s in ptb_all_tagged]
tag_seqs = [str(' '.join(t for _, t in s)) for s in ptb_all_tagged]

word_map = Tokenizer(lower=False, filters='')
word_map.fit_on_texts(texts)
print('Vocab size:', len(word_map.word_counts))

tag_map = Tokenizer(lower=False, filters='')
tag_map.fit_on_texts(tag_seqs)
print('Tag set size:', len(tag_map.word_counts))

# %%
# Split words to subwords.

def map_and_subworder(size):
    """Return a subword type map & a function from raw->subworded sentences."""
    path = 'data/ptb_all.' + size +'.txt'
    subword_texts = [s.strip('\n') for s in open(path).readlines()]

    subword_map = Tokenizer(lower=False, filters='')
    subword_map.fit_on_texts(subword_texts)

    sub = {raw: split_into_words(sub) for raw, sub in zip(texts, subword_texts)}
    for k, v in sub.items():
        assert len(k.split(' ')) == len(v), "{} vs {}".format(k, v)
    return subword_map, sub

def split_into_words(subword_sent, delimiter='@@'):
    """Take a subworded sentence and build a space-separated sequence of
    subwords for each word.

    >>> s = 'A@@ re@@ as of the fac@@ tory'
    >>> split_into_words(s)
    ['A@@ re@@ as', 'of', 'the', 'fac@@ tory']
    """
    words = []
    word = []
    for s in subword_sent.split():
        if s.endswith(delimiter):
            word.append(s)
        else:
            word.append(s)
            words.append(' '.join(word))
            word = []
    return words

subword_map, subworder = map_and_subworder('4k')

# %%
# Build a model.

num_words = len(word_map.word_index)+1
num_subwords = len(subword_map.word_index)+1
num_tags = len(tag_map.word_index)+1
max_len = max(w.count(' ') + 1 for w in subworder.keys())
max_subwords = max(w.count('@@ ') + 1 for s in subworder.values() for w in s)
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

x, y = prep(tagged_sents(ptb_train))
val = prep(tagged_sents(ptb_dev))
test = prep(tagged_sents(ptb_test))

# %%

early_stopping = EarlyStopping(monitor='val_padded_categorical_accuracy',
                               min_delta=0.001, patience=2, verbose=1)
model.fit(x, y, batch_size=32, nb_epoch=100, verbose=2, validation_data=val, callbacks=[early_stopping])
print('test_loss: {:0.4f} - test_acc: {:0.4f}'.format(*model.evaluate(*test, verbose=2)))
