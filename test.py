from hyperparameters import *
from integration_task import Integration_Task
from rnns import RNN
import torch
import torch.nn as nn
import numpy as np


# task parameters
seq_lenght = 10

task = Integration_Task()
s, t = task.generate_sample(length=seq_lenght)

print("sample:", s.shape)
print("target", t.shape)

batch_size = 3
ss = np.zeros(batch_size)
tt = np.zeros(batch_size)
for b in range(batch_size-1):
    s, t = task.generate_sample(length=seq_lenght)
    ss[b] = s
    tt[b] = t

data = [task.generate_sample(length=seq_lenght) for _ in range(batch_size)]
print(np.shape(data))

samples=data[:][0][:][:]
targets=data[:][1][:][:]
print(f"samples: {np.shape(ss)}")
