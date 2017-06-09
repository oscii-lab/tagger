"""Load a tagger and apply it."""

import sys

from keras.models import load_model
from remove_mask import *
from loss import *

custom = {
    'RemoveMask': RemoveMask,
    'padded_categorical_crossentropy': padded_categorical_crossentropy,
    'padded_categorical_accuracy': padded_categorical_accuracy,
}

print('loading model')
model = load_model(sys.argv[1], custom_objects=custom)

print('loading data')
from char_tagger import *

print('evaluating')
print('Val performance: ', model.evaluate(*val))
print('Test performance:', model.evaluate(*test))
