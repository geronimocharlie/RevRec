import torch
import numpy as np
from datetime import datetime
from rnns import *
from integration_task import Integration_Task
import pickle
from hyperparameters import *


def calculate_fixpoints(model_path, model_name, num_points):

    def generate_candidates(model_path, model_name, num_points):
        model = torch.load(model_path+model_name)
        model.eval()
        task = TASK_CLASS(batch_size=CANDIDATES_BATCHSIZE)
        candidates = None
        _i = 0
        for sample, _ in task.train_loader:
            _, hidden_states = model.forward(sample.float())
            candidates = hidden_states if candidates==None else torch.cat([candidates, hidden_states])
            if _i >CANDIDATES_ITERS:
                break
            _i = _i+1

        # save runs for trajectory plotting
        with open(model_path + 'exemplary_runs_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
            pickle.dump(candidates.detach().numpy(), file)


        candidates = torch.reshape(candidates, (-1, model.hidden_size))
        candidates = candidates[np.random.choice(CANDIDATES_SAMPLED*task.length, size=num_points, replace=False)]
        print(candidates.size())
        return candidates.detach().numpy()

    def train_fixpoints(model_path, model_name, fixpoint_candidates, stop_tol=0.0001):
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
            if loss < stop_tol:
                break
                #print(loss.size())

        # get loss (speed) fon each found fix point
        loss_f = torch.nn.MSELoss(reduction='none')
        _, hidden_states = model.forward(torch.zeros((fixpoint_candidates.size()[0],1,1)))
        fp_losses = loss_f(fixpoint_candidates, torch.squeeze(hidden_states))
        print(fp_losses.size())
        fp_losses = fp_losses.detach().numpy()
        fp_losses = np.mean(fp_losses, axis=-1)
        print(fp_losses.shape)



        with open(model_path +  'fixpoints_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
            pickle.dump((fixpoint_candidates.detach().numpy(), fp_losses), file)






    fixpoint_candidates = generate_candidates(model_path, model_name, num_points)
    trained_fixpoints = train_fixpoints(model_path, model_name, fixpoint_candidates, FP_OPT_STOP_TOL)


if __name__ == '__main__':
    MODEL_PATH = 'models/GRU_integration_09-08-2020_03-13-31_PM/'
    MODEL_NAME = 'trained_weights_GRU_integration_epochs_5'

    num_points= 5000
    calculate_fixpoints(MODEL_PATH, MODEL_NAME, num_points)
