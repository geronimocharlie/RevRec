import torch.nn as nn
import torch
import numpy as np
from hyperparameters import *
from integration_task import Integration_Task
import matplotlib.pyplot as plt

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

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          nonlinearity=activation, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)

    def init_hidden_state(self):

        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size,
                            self.hidden_size).zero_()
        return hidden

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
        out = torch.sigmoid(self.read_out(out))

        return out, hn

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size,
                            self.hidden_size).zero_()
        return hidden

    def forward(self, x, h=None, method=None):

        out, hn = self.gru(x, h)

        if method == 'last':
            out = out[:, -1, :]
        out = torch.sigmoid(self.read_out(self.relu(out)))

        return out, hn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size,
                            self.hidden_size).zero_()
        return (hidden, hidden)

    def forward(self, x, h=None, method=None):

        out, hn = self.gru(x, h)
        if method == 'last':
            out = out[:, -1, :]
        out = torch.sigmoid(self.read_out(out))

        return out, hn


def train_fn(batch_size, seq_length, num_epochs, model, task='integration', print_every=100):
    """
    @params:

    """
    losses = []
    accuracies = []

    if task == 'integration':
        task = Integration_Task(length=seq_length, batch_size=batch_size)
        task.generate_data_loader()

    # initialize optimizer
    # what does this self.parameters() refer to specifically?
    optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # loss criterion
    loss_function = nn.BCELoss()

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

    evaluate(model, task)

    for epoch in range(num_epochs):
        model.train()
        h = model.init_hidden_state()

        avg_loss = 0

        for iter, (samples, targets) in enumerate(task.train_loader):
            #samples, targets = task.generate_sample()
            #samples = torch.from_numpy(samples).float().requires_grad_()
            #targets = torch.from_numpy(targets)

            iter += 1
            h = model.init_hidden_state()
            outputs, _ = model.forward(samples.float(), h)

            loss = loss_function(outputs.float(), targets.float())

            avg_loss += loss.item()

            # gradients wrt parameters
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()

            if iter % print_every == 0:
                print(
                    f"Epoch {epoch}/{num_epochs}...Iter: {iter}/{len(task.train_loader)}....Average Loss for Epoch: {avg_loss/iter}")

        losses.append(avg_loss / iter)

        print("-------")
        print(
            f"Epoch {epoch}/{num_epochs} done, total loss: {avg_loss/len(task.train_loader)}")

        accuracies.append(evaluate(model, task))

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

    print("---------Finished Training--------")
    evaluate(model, task, last=True, losses=losses)

    return model

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


def evaluate(model, task, last=False, losses=None):
    model.eval()
    predictions = []
    targets = []
    accuracies = []

    for i, (input, target) in enumerate(task.train_loader):

        #input, target = task.generate_sample()
        #input = torch.from_numpy(input)
        #target = torch.from_numpy(target)

        h = model.init_hidden_state()

        output, h = model.forward(input.float(), h)

        output = output.detach().numpy()

        total = target.size(0) * target.size(1)
        predicted = np.round(output)
        predicted = np.squeeze(predicted)
        target = np.squeeze(target.detach().numpy())

        predictions.append(predicted)
        correct = np.sum(predicted == target)
        accuracy = 100 * (correct / total)
        accuracies.append(accuracy)

    if last & (losses != None):
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(accuracies)
        axes[1].plot(losses)
        plt.show()

    print(f"Mean Accuracy: {np.mean(accuracies)}")

    return accuracies


if __name__ == "__main__":
    input_size = 1
    hidden_size = 256
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

    output_size = 1
    length = 100
    batch_size = 5
    num_epochs = 200
    model = GRU(input_size, hidden_size, num_layers, output_size, batch_size)
    #model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
    train_fn(batch_size, length, num_epochs, model, 'integration')
