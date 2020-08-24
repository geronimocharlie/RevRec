import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader
import torch


class Flipflop_task():
    class Channel():
        def __init__(self, change_probability):
            self.possble_inputs = [1., -1.]
            self.change_probability = change_probability

        def create_trajectory(self, length=100):
            initial_input = random.choice(self.possble_inputs)
            inputs = [initial_input]
            targets = [initial_input]
            current_target = initial_input
            for _ in range(length-1):
                if random.random() < self.change_probability:
                    current_target = current_target*-1
                    current_input = current_target
                else:
                    current_input = 0.
                inputs.append(current_input)
                targets.append(current_target)
            return np.asarray(inputs), np.asarray(targets)



    def __init__(self, channel_probs=[.1,.1,.1], length=100, num_samples=10000, batch_size=10):
        """
        @params:
            - channel_probs: list of len num_channels, respective entry is the probability of a change in a single step
        """
        self.channels = [self.Channel(change_prob) for change_prob in channel_probs]
        self.length = length
        self.batch_size = batch_size
        self.num_samples=num_samples
        self.size = len(channel_probs)

    def generate_sample(self, length):
        input = []
        target = []
        for channel in self.channels:
            c_in, c_tar = channel.create_trajectory()
            input.append(c_in)
            target.append(c_tar)
        input = np.stack(input, axis=-1)
        target = np.stack(target, axis=-1)
        return input, target

    def generate_samples(self, num_samples, length=None):
        inputs = []
        targets = []
        for _ in range(num_samples):
            s_in, s_tar = self.generate_sample(length)
            inputs.append(s_in)
            targets.append(s_tar)
        inputs = np.stack(inputs)
        targets = np.stack(targets)
        targets[targets==-1]=0
        return inputs, targets

    def generate_data_loader(self, length=None, num_samples=None, batch_size=None):

        length = (length or self.length)
        num_samples = (num_samples or self.num_samples)
        batch_size = (batch_size or self.batch_size)

        inputs, targets = self.generate_samples(num_samples, length)

        train_portion = int(0.9 * inputs.shape[0])

        train_data = TensorDataset(torch.from_numpy(
            inputs[0:train_portion]), torch.from_numpy(targets[0:train_portion]))
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)

        test_data = TensorDataset(torch.from_numpy(
            inputs[train_portion:]), torch.from_numpy(targets[train_portion:]))
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)

        return train_loader, test_loader

if __name__ == '__main__':
    test = Flipflop_task([.1,.1,.1])
    samples = test.generate_samples(10000)
    print(samples)
