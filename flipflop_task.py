import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader


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



    def __init__(self, channel_probs, length=100, batch_size=32):
        """
        @params:
            - channel_probs: list of len num_channels, respective entry is the probability of a change in a single step
        """
        self.channels = [Channel(change_prob) for change_prob in channel_probs]
        self.length = length
        self.batch_size = batch_size,

    def generate_sample(self, length):
        input = []
        target = []
        for channel in self.channels:
            c_in, c_tar = channel.create_trajectory
            input.append(c_in)
            target.append(c_tar)
        input = np.stack(input, axis=-1)
        target = np.stack(target, axis=-1)
        return input, target

    def generate_samples(num_samples, length=None):
        inputs = []
        targets = []
        for _ in range(num_samples):
            s_in, s_tar = self.generate_sample(length)
            inputs.append(s_in)
            targets.append(s_tar)
        inputs = np.stack(inputs)
        targets = np.stack(targets)
        return inputs, targets

    def generate_data_loader(self, size, length=100):

        inputs, targets = self.generate_samples(size, length)
        train_data = TensorDataset(torch.from_numpy(
            inputs), torch.from_numpy(targets))
        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)

        test_data = TensorDataset(torch.from_numpy(
            test_x), torch.from_numpy(test_y))
        self.test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
