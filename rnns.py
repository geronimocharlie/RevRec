import torch.nn as nn
import torch
import numpy as np


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, activation='tanh'):
        super(RNN, self).__init__()
        '''
        @params:
            input_size [int]:  expected features in input
            hidden_size [int]: numbers of features in hidden state
            num_layers [int]: number of recurrent layers
            activation [str]: activation function

        '''
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=activation, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)

        self.h0 = torch.zeros(num_layers, batch_size, hidden_size)


    def forward(self, x, h=None):
        """
        @params:
            x: (batch_size, lenght, input_size)
            h: hidden state (num_layers, batch_size, hidden_size)
        """
        if h is None: h=self.h0
        # wrap hidden state in a fresh variable: construct new view
        out, hn = self.rnn(x, h.detach())

        # only last outs : (batch_size, input_size)
        out_last = self.read_out(out[:, -1, :])

        return out_last, hn


if __name__=="__main__":
    input_size = 10
    hidden_size = 5
    num_layers = 2
    output_size = 1
    length = 5
    batch_size = 1
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)

    i = torch.randn(batch_size, length, input_size)
    print("h0:", model.h0.size())

    o, h = model.forward(i)
    print("h1:", model.h0.size())
    print("out:", o)
    assert ~torch.all(model.h0.eq(h)), "sth wrong with forward step"
