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
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          nonlinearity=activation, batch_first=True)
        # Fully connected layer
        self.read_out = nn.Linear(hidden_size, output_size)

        # This generates the first hidden state of zeros that will be used in the forward pass
        self.h0 = torch.zeros(num_layers, batch_size, hidden_size)

        # dim=-1??
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, h=None):
        """
        @params:
            x: (batch_size, length, input_size)
            h: hidden state (num_layers, batch_size, hidden_size)
        """
        if h is None:
            h = self.h0
        # print(type(x))
        #x = x.astype(float)
        x = x.float()
        # wrap hidden state in a fresh variable: construct new view
        # is .detach() used to avoid backpropagating all the way to the start?
        out, hn = self.rnn(x, h.detach())

        # only last outs : (batch_size, input_size)
        # out_last = self.read_out(out[:, -1, :])
        out = self.softmax(self.read_out(out))

        return out, hn

    def train(self, batch_size, seq_length, num_epochs, task='integration', print_every=100):
        """
        @params:

        """
        #num_epochs = n_iters / (len(samples)/batch_size)
        #num_epochs = int(num_epochs)

        if task == 'integration':
            task = Integration_Task(length=seq_length, batch_size=batch_size)

        # initialize optimizer
        # what does this self.parameters() refer to specifically?
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # loss criterion
        loss_function = nn.CrossEntropyLoss()

        iter = 0
        for epoch in range(num_epochs):
            # self.rnn.train()
            samples, targets = task.generate_sample()
            samples = torch.from_numpy(samples)
            targets = torch.from_numpy(targets)

            # samples with gradient accumulation abilities
            # why is it with gradient accumulation abilities? Is view function meant to reshape the tensor?
            samples = samples.view(samples.size())
            # clear gradients
            optimizer.zero_grad()

            outputs, _ = self.forward(samples)

            #print(f"original target shape: {targets.size()}")
            #print(f"original output shape: {outputs.size()}")

            #print(f"target view: {targets.view(batch_size, -1).size()}")
            #print(f"output view: {outputs.view(batch_size, -1, seq_length).size()}")
            #print(f"target view: {targets.view(-1, batch_size*seq_length).size()}")

            # calculate loss
            #loss = loss_function(outputs, targets)
            #loss = loss_function(outputs.view(-1, batch_size*seq_length), targets.contiguous().view(-1))
            loss = loss_function(outputs.view(
                batch_size, -1, seq_length), targets.view(batch_size, -1))
            # accumulate gradient for each parameter
            loss.backward()
            # update parameters based on the current gradient
            optimizer.step()

            # iter+=1

            if epoch % print_every == 0:
                # self.rnn.eval()

                correct = 0
                total = 0
                samples, targets = task.generate_sample()
                samples = torch.from_numpy(samples)
                targets = torch.from_numpy(targets)

                # samples with gradient accumulation abilities
                samples = samples.view(samples.size()).requires_grad_()

                # what are the dimensions?
                _, predicted = torch.max(outputs.data, 2)
                print(predicted.size(), "predicted")

                total = targets.size(0) * targets.size(1)
                print(f"total: {total}")
                # don't understand the np.sum and np.squeeze part
                correct = np.sum(np.squeeze(predicted.numpy())
                                 == np.squeeze(targets.numpy()))
                print(f"correct: {correct}")

                accuracy = 100 * correct / total
                print(
                    f"Iteration: {iter}. Loss: {loss}. Accuracy: {accuracy}.")


if __name__ == "__main__":
    input_size = 1
    hidden_size = 128
    num_layers = 1
    output_size = 2  # the google github: number of outputs is 1
    length = 100  # what is sequence length in the integration task?
    batch_size = 50
    num_epochs = 10000
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)

    i = torch.randn(batch_size, length, input_size)
    print("h0:", model.h0.size())

    #o, h = model.forward(i)
    #print("h1:", model.h0.size())
    #print("out:", o)
    #assert ~torch.all(model.h0.eq(h)), "sth wrong with forward step"

    model.train(batch_size, length, num_epochs, 'integration')
