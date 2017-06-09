'''Layer to remove masking.

https://gist.github.com/udibr/676c742c8843fdcfdfd24f4dcdc3bdfb
'''

from keras.layers import Lambda

class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None
