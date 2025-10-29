import numpy as np
import torch


a = torch.tensor([1, 2, 3])
print(a)

s = a.sum()
print(s)

print(s.item())

print(torch.tensor(1))