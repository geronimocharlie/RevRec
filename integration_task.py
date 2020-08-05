from scipy import signal
import numpy as np
"""
Class that generates samples for a timeseries integration task
"""
class Integration_Task():
    """
    @params:
     - discount: discount factor for exponentially discounted sum
     - proto_length: default length of a single sample, if no other length is specified in the sampling statement
     - loc: mean of the noise used as input
     - scale: scale of the noise used as input
    """
    def __init__(self, discount=1., length=100, size=1, batch_size=1, loc=0, scale=1.):
        self.length = length
        self.size = size
        self.batch_size = batch_size
        self.discount = discount
        self.loc = loc
        self.scale = scale

    def generate_sample(self, length=None, size=None, batch_size=None, discount=None, loc=None, scale=None):
        """
        Function that samples an input target timesieres pair
        """
        length = (length or self.length)
        size = (size or self.size)
        batch_size = (batch_size or self.batch_size)
        shape = (batch_size, length, size)
        discount = (discount or self.discount)
        loc = (loc or self.loc)
        scale = (scale or self.scale)


        sample = np.random.normal(loc=loc, scale=scale, size=shape)
        target = (discount_cumsum(sample, discount) > 0).astype(np.int)
        return sample, target


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=1)[::-1]
