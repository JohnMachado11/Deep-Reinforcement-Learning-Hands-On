import numpy as np
import torch


a = torch.FloatTensor([2, 3])
print(a)

# ca = a.to("cuda") # Not ony Mac
ca = a.to("mps") # On Mac
print(ca) # tensor([2., 3.], device='mps:0') device='mps:0', 0 means the first GPU card on the system. The index is zero-based.

print(ca + 1)
print(ca.device)