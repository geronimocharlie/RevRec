import torch.nn as nn
import torch
import numpy as np
from hyperparameters import *
from integration_task import Integration_Task
import matplotlib.pyplot as plt
from datetime import datetime
import os

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
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=activation, batch_first=True)
        self.read_out = nn.Linear(hidden_size, output_size)
        self.name = 'RNN'
        model.folder_name = ""


    def init_hidden_state(self):

        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()
        return hidden


    def forward(self, x, h=None):
        """
        @params:
            x: (batch_size, lenght, input_size)
            h: hidden state (num_layers, batch_size, hidden_size)
        """

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

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.read_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.name = 'GRU'
        self.folder_name = ""

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()
        return hidden


    def forward(self, x, h=None, method=None):


        out, hn = self.gru(x, h)

        if method == 'last':
            out = out[:,-1,:]
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

        self.gru = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.read_out = nn.Linear(hidden_size, output_size)

        self.name = 'LSTM'
        self.folder_name = ""

    def init_hidden_state(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_()
        return (hidden, hidden)

    def forward(self, x, h=None, method=None):

        out, hn = self.gru(x, h)
        if method == 'last':
            out = out[:,-1,:]
        out = torch.sigmoid(self.read_out(out))

        return out, hn



def train_fn(batch_size, seq_length, num_epochs, model, task_name='integration', print_every=100, path = 'models/'):
        """
        @params:

        """
        losses = []
        accuracies = []


        if task_name=='integration':
            task = Integration_Task(length=seq_length, batch_size=batch_size)
            task.generate_data_loader()



        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # loss criterion
        loss_function = nn.BCELoss()

        evaluate(model, task)

        for epoch in range(num_epochs):
            epoch += 1
            model.train()
            h = model.init_hidden_state()

            avg_loss = 0

            for iter , (samples, targets) in enumerate(task.train_loader):
                #samples, targets = task.generate_sample()
                #samples = torch.from_numpy(samples).float().requires_grad_()
                #targets = torch.from_numpy(targets)


                iter +=1
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
                    print(f"Epoch {epoch}/{num_epochs}...Iter: {iter}/{len(task.train_loader)}....Average Loss for Epoch: {avg_loss/iter}")

            losses.append(avg_loss/iter)

            print("-------")
            print(f"Epoch {epoch}/{num_epochs} done, total loss: {np.mean(losses)}")

            accuracies.append(evaluate(model, task))




        print("---------Finished Training--------")
        evaluate(model, task, last=True, avg_losses=losses, avg_accuracies=accuracies)
        save(path, model, task_name, epoch)


def save(supfolder, model, task_name, epoch):

    time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    model.folder_name = f"{supfolder}/{model.name}_{task_name}_{time_stamp}"
    os.mkdir(model.folder_name)
    torch.save(model, f"{model.folder_name}__epochs_{epoch}")

def evaluate(model, task, last=False, avg_losses=None, avg_accuracies=None):
    model.eval()
    predictions = []
    targets = []
    accuracies = []

    for i ,(input, target) in enumerate(task.train_loader):

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
        correct = np.sum(predicted==target)
        accuracy = 100 * (correct / total)
        accuracies.append(accuracy)

    if last & (avg_losses!=None) & (avg_accuracies!=None):
        fig, axes = plt.subplots(2,1)
        axes[0].plot(avg_accuracies)
        axes[1].plot(avg_losses)
        plt.show()

    print(f"Mean Accuracy: {np.mean(accuracies)}")

    return np.mean(accuracies)

def load_last(folder="models/"):
    pass



if __name__=="__main__":
    input_size = 1
    hidden_size = 256
    num_layers = 1
    output_size = 1
    length = 10
    batch_size = 5
    num_epochs = 2
    model = GRU(input_size, hidden_size, num_layers, output_size, batch_size)
    #model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
    train_fn(batch_size, length, num_epochs, model, 'integration')

    model = torch.load
