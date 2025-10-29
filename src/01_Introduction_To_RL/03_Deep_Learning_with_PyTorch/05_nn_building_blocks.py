import numpy as np
import torch.nn as nn
import torch

# Random weights every time
l = nn.Linear(2, 5)
print(l)

v = torch.FloatTensor([1, 2])
print(v.shape)
print(l(v))

print("--------------------------------")
# Sequential
s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)

print(s)

print("\nNeural Net Output:\n")
print(s(torch.FloatTensor([[1, 2]])))