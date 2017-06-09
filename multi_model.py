'''Multi-model hack for Keras.'''

import functools

from keras.models import Model


class MultiModel(Model):
    """A wrapper that chooses a model based on method input x.

    example_model -- The model to use when the input is unknown.
    choose_model  -- A fn from x (an input tensor) to a model.
    """
    def __init__(self, example_model, choose_model):
        self.example_model = example_model
        self.choose_model = choose_model

        # If an attribute is not found on the example, look in this object
        example_model.__getattr__ = self.__getattribute__

    def __getattribute__(self, name):
        """Return instance attribute, or x-specific attr, or example attr."""
        if name == '__dict__' or name in self.__dict__.keys():
            return object.__getattribute__(self, name)

        a = self.example_model.__getattribute__(name)
        if hasattr(a, '__func__'): # It's a method!
            varnames = a.__func__.__code__.co_varnames
            if 'x' in varnames:
                x_pos = varnames.index('x')

                @functools.wraps(a)
                def late_binding_method(*args, **vargs):
                    x = vargs['x'] if 'x' in vargs else args[x_pos]
                    x_specific = self.choose_model(x)

                    # If an attribute is not found, look in this object
                    x_specific.__getattr__ = self.__getattribute__

                    return x_specific.__getattribute__(name)(*args, **vargs)

                return late_binding_method
            else:
                return object.__getattribute__(self, name)
        return a
