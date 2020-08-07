from scipy import signal
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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
        @output:
            sample: (batch_size, lenght, size)
            target: (batch_size, lenght, out_size)
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

    def generate_data_loader(self, length=None, size=None, batch_size=None, data_size=10000, discount=None, loc=None, scale=None, method=None):

        length = (length or self.length)
        size = (size or self.size)
        batch_size = (batch_size or self.batch_size)
        discount = (discount or self.discount)
        loc = (loc or self.loc)
        scale = (scale or self.scale)
        shape = (data_size, length, size)
        #method = (method or self.method)
        samples = np.random.normal(loc=loc, scale=scale, size=shape)

        if method == 'last':
            targets = (np.sum(samples, axis=1)>0).astype(np.int)
            #print(f"sample: {samples} {samples.shape}")
            #print(f"target: {targets} {targets.shape}")

        else:
            targets = (discount_cumsum(samples, discount) > 0).astype(np.int)
        test_portion = int(0.1*len(samples))
        train_x = samples[:-test_portion]
        train_y = targets[:-test_portion]
        test_x = samples[-test_portion:]
        test_y = targets[-test_portion:]
        print(f"train x: {train_x.shape}")
        print(f"train y: {train_y.shape}")
        print(f"test x: {test_x.shape}")
        print(f"test y: {test_y.shape}")


        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        self.test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
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

if __name__=="__main__":
    task = Integration_Task()
    task.generate_data_loader()


    #for i, (x, target) in enumerate(task.train_loader):
        #print(i, "i")
        #print(x.size(), "x")
        #print(target.size(), "target")

    print("------------------------------")

    for i, (x, target) in enumerate(task.test_loader):
        print(i, "i")
        print(x.size(), "x")
        print(target.size(), "target")
