import torch
import rnns
from integration_task import Integration_Task
CANDIDATES_BATCHSIZE = 50
task_class = Integration_Task
model = rnns.RNN

def calculate_fixpoints(model_path, model_name, num_points):

    def generate_candidates(model_path, model_name, num_points):
        model = torch.load(model_path)
        task = task_class(batch_size=CANDIDATES_BATCHSIZE)
        _, hidden_states = model.forward()
    def train_fixpoints(model_path, fixpoint_candidates):

    fixpoint_candidates = generate_candidates(model_path, num_points)
    trained_fixpoints = train_fixpoints(model_path, fixpoint_candidates)
