import torch.nn as nn
import tocrh
import numpy as np


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, activation='tanh', output_size):
        '''
        @params:
            input_size [int]:  expected features in input
            hidden_size [int]: numbers of features in hidden state
            num_layers [int]: number of recurrent layers
            activation [str]: activation function

        '''
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=activation, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)

        self.h0 = tocrh.zeros(num_layers, batch_size, hidden_size)

    def forward(self, x, h=self.h0):
        """
        @params:
            x: (batch_size, lenght, input_size)
        """

        # wrap hidden state in a fresh variable: construct new view
        out, hn = self.rnn(x, h.detach())

        # only last outs : (batch_size, input_size)
        out_last = self.fc(out[:, -1, :])

        return out_last, hn
