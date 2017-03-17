"""Read a version of a corpus that has split words to subwords."""

from keras.preprocessing.text import Tokenizer

def map_and_subworder(texts, subword_paths, max_subwords):
    """Return a subword type map & a dict from raw->subworded sentences."""
    lines = [s for p in subword_paths for s in open(p).readlines()]
    sub_texts = [s.strip('\n') for s in lines]
    assert len(texts) == len(sub_texts), (len(texts), len(sub_texts))

    subword_map = Tokenizer(lower=False, filters='')
    subword_map.fit_on_texts(sub_texts)

    sub = {raw: split_into_words(sub, max_subwords) for raw, sub in zip(texts, sub_texts)}
    for k, v in sub.items():
        assert len(k.split(' ')) == len(v), "{} vs {}".format(k, v)
    return subword_map, sub

def split_into_words(subword_sent, max_subwords, delimiter='@@'):
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
            words.append(' '.join(word[:max_subwords]))
            word = []
    return words