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
        print(type(x))
        #x = x.astype(float)
        x = x.float()
        # wrap hidden state in a fresh variable: construct new view
        out, hn = self.rnn(x, h.detach())

        # only last outs : (batch_size, input_size)
        # out_last = self.read_out(out[:, -1, :])
        out = self.read_out(out)

        return out, hn

    def train(self, batch_size, n_iters, seq_length, num_epochs, task='integration', print_every=1):
        """
        @params:

        """
        #num_epochs = n_iters / (len(samples)/batch_size)
        #num_epochs = int(num_epochs)

        if task=='integration':
            task = Integration_Task(length=seq_length, batch_size=batch_size)



        # initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # loss criterion
        loss_function = nn.CrossEntropyLoss()


        iter = 0
        for epoch in range(num_epochs):
            self.rnn.train()
            samples, targets = task.generate_sample()
            samples = torch.from_numpy(samples)
            targets = torch.from_numpy(targets)


            # samples with gradient accumulation abilities
            samples = samples.view(samples.size()).requires_grad_()
            # clear gradients
            optimizer.zero_grad()

            outputs, _ = self.forward(samples)

            print(f"original target shape: {targets.size()}")
            print(f"original output shape: {outputs.size()}")

            print(f"target view: {targets.view(batch_size, -1).size()}")
            print(f"output view: {outputs.view(batch_size, -1, seq_length).size()}")
            #print(f"target view: {targets.view(-1, batch_size*seq_length).size()}")

            # calculate loss
            #loss = loss_function(outputs, targets)
            #loss = loss_function(outputs.view(-1, batch_size*seq_length), targets.contiguous().view(-1))
            loss = loss_function(outputs.view(batch_size, -1, seq_length), targets.view(batch_size, -1))
            # gradients wrt parameters
            loss.backward()
            # update parameters
            optimizer.step()

            iter+=1

            if iter % print_every == 0:
                self.rnn.eval()

                correct = 0
                total = 0
                samples, targets = task.generate_sample()
                outputs, _ = self.forward(samples)

                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)

                correct += (predicted == targets).sum()

            accuracy = 100 * correct / total
            print(f"Iteration: {iter}. Loss: {loss}. Accuracy: {accuracy}.")



if __name__=="__main__":
    input_size = 1
    hidden_size = 5
    num_layers = 2
    output_size = 2
    length = 5
    batch_size = 1
    n_iters = 10
    num_epochs = 10
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)

    i = torch.randn(batch_size, length, input_size)
    print("h0:", model.h0.size())

    #o, h = model.forward(i)
    #print("h1:", model.h0.size())
    #print("out:", o)
    #assert ~torch.all(model.h0.eq(h)), "sth wrong with forward step"

    model.train(batch_size, n_iters, length, num_epochs, 'integration')
