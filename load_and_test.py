"""Load a tagger and apply it."""

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
from char_tagger import *

# %%

from keras.models import load_model

model = load_model(sys.argv[1])
print('Test performance:', model.evaluate(*test))
