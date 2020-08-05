import torch.nn as nn
import torch
import numpy as np
from hyperparameters import *
from integration_task import Integration_Task

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

    def train(self, batch_size, n_iters, seq_length, num_epochs, task='integration'):
        """
        @params:
            samples: (how_many, seq_lenght, input_size)
            targets: (how_many, seq_lenght, out_size)
        """
        #num_epochs = n_iters / (len(samples)/batch_size)
        #num_epochs = int(num_epochs)

        if task=='integration':
            task = Integration_Task(proto_length=seq_length)


        # initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # loss criterion
        loss_function = nn.CrossEntropyLoss()

        iter = 0
        for epoch in range(num_epochs):
            self.rnn.train()
            samples, targets = task.generate_sample_batch(batch_size)
            samples = torch.from_numpy(samples)
            targets = torch.from_numpy(targets)

            # samples with gradient accumulation abilities
            samples = samples.view(samples.size()).requires_grad_()
            # clear gradients
            optimizer.zero_grad()

            outputs, _ = self.forward(samples)

            # calculate loss
            loss = loss_function(outputs, targets)
            # gradients wrt parameters
            loss.backward()
            # update parameters
            optimizer.step()

            iter+=1

            if iter % print_every == 0:
                self.rnn.eval()

                correct = 0
                total = 0
                samples, targets = task.generate_sample_batch(batch_size)
                outputs, _ = self.forward(samples)

                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)

                correct += (predicted == targets).sum()

            accuracy = 100 * correct / total
            print(f"Iteration: {iter}. Loss: {loss}. Accuracy: {accuracy}.")



if __name__=="__main__":
    input_size = 10
    hidden_size = 5
    num_layers = 2
    output_size = 1
    length = 5
    batch_size = 1
    n_iters = 10
    num_epochs = 10
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)

    i = torch.randn(batch_size, length, input_size)
    print("h0:", model.h0.size())

    o, h = model.forward(i)
    print("h1:", model.h0.size())
    print("out:", o)
    assert ~torch.all(model.h0.eq(h)), "sth wrong with forward step"

    model.train(batch_size, n_iters, length, num_epochs, 'integration')
