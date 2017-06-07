"""Load Part-of-speech tagging data sets."""

from keras.preprocessing.text import Tokenizer

from nltk.corpus import BracketParseCorpusReader as reader

# %%
# Load data.

# Create a data directory and then unarchive the following:
# - https://catalog.ldc.upenn.edu/LDC2015T13 (Penn Treebank Revised)
# - https://catalog.ldc.upenn.edu/LDC2012T13 (English Web Treebank)

# Standard part-of-speech tagging train/dev/test split
#ptb_root = 'data/eng_news_txt_tbnk-ptb_revised/data/penntree'
#ptb_train = reader(ptb_root, r'(0[0-9]|1[0-8])/.*tree')
#ptb_dev = reader(ptb_root, r'(19|20|21)/wsj_.*tree')
#ptb_test = reader(ptb_root, r'(22|23|24)/wsj_.*tree')
#ptb_all = reader(ptb_root, r'[0-9]*/wsj_.*tree')
ptb_root = 'data/eng_news_txt_tbnk-ptb'
ptb_train = reader(ptb_root, r'(0[0-9]|1[0-8])/.*mrg')
ptb_dev = reader(ptb_root, r'(19|20|21)/wsj_.*mrg')
ptb_test = reader(ptb_root, r'(22|23|24)/wsj_.*mrg')
ptb_all = reader(ptb_root, r'[0-9]*/wsj_.*mrg')

web_root = 'data/eng_web_tbk/data'
web_genres = ['answers', 'email', 'newsgroup', 'reviews', 'weblog']
web_all = [reader(web_root, g + r'/penntree/.*tree') for g in web_genres]

# %%
# Index all types as integers.

def tagged_sents(corpora):
    """Iterate over tagged sentences, dropping fake words such as *PRO*."""
    for corpus in corpora:
        for ts in corpus.tagged_sents():
            yield [(w, t) for w, t in ts if t != '-NONE-']

all_tagged = list(tagged_sents([ptb_all] + list(web_all)))
texts = [str(' '.join(w for w, _ in s)) for s in all_tagged]
tag_seqs = [str(' '.join(t for _, t in s)) for s in all_tagged]

word_map = Tokenizer(lower=False, filters='')
word_map.fit_on_texts(texts)

tag_map = Tokenizer(lower=False, filters='')
tag_map.fit_on_texts(tag_seqs)
