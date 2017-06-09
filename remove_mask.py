'''Layer to remove masking.

Modified version of
https://gist.github.com/udibr/676c742c8843fdcfdfd24f4dcdc3bdfb
'''

from keras.layers import Lambda

class RemoveMask(Lambda):
    def __init__(self, **kwargs):
        kwargs['function'] = lambda x, mask: x
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None
