import torch
import rnns
from integration_task import Integration_Task
CANDIDATES_BATCHSIZE = 50
CANDIDATES_ITERS = 50
task_class = Integration_Task
model = rnns.RNN

def calculate_fixpoints(model_path, model_name, num_points):

    def generate_candidates(model_path, model_name, num_points):
        model = torch.load(model_path+model_name)
        model.eval()
        task = task_class(batch_size=CANDIDATES_BATCHSIZE)
        candidates = None
        _i = 0
        for sample, _ in task.train_loader:
            _, hidden_states = model.forward(sample)
            candidates = hidden_states if candidates==None else torch.cat([candidates, hidden_states])
            if _i >=CANDIDATES_ITERS:
                break
            _i = _i+1
        print(candidates.shape)

        print(hidden_states.size)
    def train_fixpoints(model_path, fixpoint_candidates):

    fixpoint_candidates = generate_candidates(model_path, num_points)
    trained_fixpoints = train_fixpoints(model_path, fixpoint_candidates)


if __name__ == '__main__'
