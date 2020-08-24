import torch
import numpy as np
from datetime import datetime
from rnns import *
from integration_task import Integration_Task
import pickle
from hyperparameters import *

#TASK_CLASS = Integration_Task


def generate_candidates(model_path, model_name, num_points):
    model = torch.load(model_path+model_name)
    model.eval()
    task = TASK_CLASS(batch_size=CANDIDATES_BATCHSIZE)
    candidates = None
    _i = 0

    train_loader = task.generate_data_loader()[0]
    for sample, _ in train_loader:
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

def train_fixpoints(model_path, model_name, fixpoint_candidates, stop_tol=0.0001, input='zeros'):
    task = TASK_CLASS(batch_size=CANDIDATES_BATCHSIZE)
    fixpoint_candidates = torch.from_numpy(fixpoint_candidates)
    fixpoint_candidates.requires_grad_()
    if input=='zeros':
    #    x_star = torch.zeros((fixpoint_candidates.size()[0],1,1))
        x_star = torch.zeros((fixpoint_candidates.size()[0],1,task.size))
    if input=='negative':
        x_star = torch.add(torch.zeros((fixpoint_candidates.size()[0],1,task.size)), -1)
    if input=='positive':
        x_star = torch.add(torch.zeros((fixpoint_candidates.size()[0],1,task.size)), 1)

    model = torch.load(model_path+model_name)
    loss_f = torch.nn.MSELoss()
    optimizer=torch.optim.Adam([fixpoint_candidates], LEARNING_RATE)
    optimizer.zero_grad()
    for i in range(OPTIMIZATION_ITERS):
        _, hidden_states = model.forward(x_star)
        loss = loss_f(fixpoint_candidates, torch.squeeze(hidden_states))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%PRINT_EVERY == 0:
            print(loss)
        if loss < stop_tol:
            break
    #with open(model_path +  'fixpoints_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
    #    pickle.dump(fixpoint_candidates.detach().numpy(), file)

    # get loss (speed) fon each found fix point
    loss_f = torch.nn.MSELoss(reduction='none')
    _, hidden_states = model.forward(torch.zeros((fixpoint_candidates.size()[0],1,task.size)))
    fp_losses = loss_f(fixpoint_candidates, torch.squeeze(hidden_states))
    print(fp_losses.size())
    fp_losses = fp_losses.detach().numpy()
    fp_losses = np.mean(fp_losses, axis=-1)
    print(fp_losses.shape)



    with open(model_path +  'fixpoints_and_losses_input_' + input + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
        pickle.dump((fixpoint_candidates.detach().numpy(), fp_losses), file)

def save_runs_artificial_input(model_path, model_name, batch_size, mode='increasing'):
        model = torch.load(model_path+model_name)
        model.eval()
        task = TASK_CLASS(batch_size = batch_size)
        runs = None
        sample, _ = task.generate_fix_sample()
        _, runs = model.forward(sample.float())

        with open(model_path + f'{mode}_runs_' + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"), 'wb') as file:
            pickle.dump(runs.detach().numpy(), file)


if __name__ == '__main__':
    model_path = 'models/GRU_integration_09-08-2020_03-13-31_PM/'
    model_name = 'trained_weights_GRU_integration_epochs_5'

    num_points= 10000
    variants= ['zeros', 'positive', 'negative']
    fixpoint_candidates = generate_candidates(MODEL_PATH, MODEL_NAME, num_points)
    for v in variants:
        train_fixpoints(MODEL_PATH, MODEL_NAME, fixpoint_candidates, FP_OPT_STOP_TOL, input=v)
    #save_runs_artificial_input(MODEL_PATH, MODEL_NAME, 20, mode='decreasing')
    #save_runs_artificial_input(MODEL_PATH, MODEL_NAME, 20, mode='increasing')
