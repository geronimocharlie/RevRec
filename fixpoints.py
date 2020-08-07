import torch
import numpy as np
from datetime import datetime
from rnns import *
from integration_task import Integration_Task
CANDIDATES_BATCHSIZE = 50
CANDIDATES_ITERS = 50
#amount of of sampled possible pre-candidates, num_points must be smaller than this.
CANDIDATES_SAMPLED=CANDIDATES_BATCHSIZE*CANDIDATES_ITERS
P_NORM=2
LEARNING_RATE = 0.001
OPTIMIZATION_ITERS = 10000
PRINT_EVERY = 1000
import pickle
task_class = Integration_Task

def calculate_fixpoints(model_path, model_name, num_points):

    def generate_candidates(model_path, model_name, num_points):
        model = torch.load(model_path+model_name)
        model.eval()
        task = task_class(batch_size=CANDIDATES_BATCHSIZE)
        candidates = None
        _i = 0
        for sample, _ in task.train_loader:
            _, hidden_states = model.forward(sample.float())
            candidates = hidden_states if candidates==None else torch.cat([candidates, hidden_states])
            if _i >CANDIDATES_ITERS:
                break
            _i = _i+1
        candidates = torch.reshape(candidates, (-1, model.hidden_size))
        candidates = candidates[np.random.choice(CANDIDATES_SAMPLED*task.length, size=num_points, replace=False)]
        print(candidates.size())
        return candidates.detach().numpy()

    def train_fixpoints(model_path, model_name, fixpoint_candidates):
        fixpoint_candidates = torch.from_numpy(fixpoint_candidates)
        fixpoint_candidates.requires_grad_()
        model = torch.load(model_path+model_name)
        loss_f = torch.nn.MSELoss()
        optimizer=torch.optim.Adam([fixpoint_candidates], LEARNING_RATE)
        optimizer.zero_grad()
        for i in range(OPTIMIZATION_ITERS):
            _, hidden_states = model.forward(torch.zeros((fixpoint_candidates.size()[0],1,1)))
            loss = loss_f(fixpoint_candidates, torch.squeeze(hidden_states))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%PRINT_EVERY == 0:
                print(loss)
        with open(model_path +  'fixpoints_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
            pickle.dump(fixpoint_candidates.detach().numpy(), file)





    fixpoint_candidates = generate_candidates(model_path, model_name, num_points)
    trained_fixpoints = train_fixpoints(model_path, model_name, fixpoint_candidates)


if __name__ == '__main__':
    MODEL_PATH = '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/'
    MODEL_NAME = 'trained_weights_GRU_integration_epochs_5'

    num_points= 800
    calculate_fixpoints(MODEL_PATH, MODEL_NAME, num_points)
