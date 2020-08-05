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
    def __init__(self, discount=1., proto_length=100, loc=0, scale=1.):
        self.discount = discount
        self.proto_length = proto_length
        self.loc = loc
        self.scale = scale

    def generate_sample(self, length=None, discount=None, loc=None, scale=None):
        """
        Function that samples an input target timesieres pair
        """
        length = (length or self.proto_length)
        discount = (discount or self.discount)
        loc = (loc or self.loc)
        scale = (scale or self.scale)


        sample = np.expand_dims(np.random.normal(loc=loc, scale=scale, size=length), axis=-1)
        target = discount_cumsum(sample, discount)
        return sample, target

    def generate_sample_batch(self, batch_size=1, length=None, discount=None, loc=None, scale=None):
        """
        @out:
            samples: (batch_size, seq_lenght, input_size)
            targets: (batch_size, seq_lenght, out_size)
        """
        samples = []
        targets = []
        for _ in range(batch_size):
            s, t = self.generate_sample(length, discount, loc, scale)
            samples.append(s)
            targets.append(t)
        return np.array(samples), np.array(targets)


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
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
