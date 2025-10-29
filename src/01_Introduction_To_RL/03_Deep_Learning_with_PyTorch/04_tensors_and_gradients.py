import numpy as np
import torch

# Create a tensor v1 that requires gradients (so PyTorch tracks all operations done on it)
v1 = torch.tensor([1.0, 1.0], requires_grad=True)

# Create another tensor v2 that does NOT require gradients
v2 = torch.tensor([2.0, 2.0])

# Perform element-wise addition: v_sum = v1 + v2
# Since v1 requires gradients, v_sum will also require gradients
v_sum = v1 + v2
print(v_sum)  # tensor([3., 3.], grad_fn=<AddBackward0>)

# Multiply v_sum by 2, then sum all elements to get a single scalar
# This scalar is the final result we’ll use to compute gradients
v_res = (v_sum * 2).sum()
print(v_res)  # tensor(12., grad_fn=<SumBackward0>)

print("----------------")

# .is_leaf tells whether a tensor is a leaf node in the computation graph
# A leaf tensor is one that was created by the user (not the result of another operation)
print("v1 is leaf:", v1.is_leaf)   # True → created directly by user
print("v2 is leaf:", v2.is_leaf)   # True → created directly by user

# .requires_grad shows whether PyTorch is tracking operations for gradient computation
print("v1 requires grad:", v1.requires_grad)  # True
print("v2 requires grad:", v2.requires_grad)  # False
print("v_sum requires grad:", v_sum.requires_grad)  # True (inherits from v1)
print("v_res requires grad:", v_res.requires_grad)  # True (depends on v_sum)

# Backpropagate from v_res through the computation graph
# This computes d(v_res)/d(v1)
print(v_res.backward())

# v_res is the final scalar result (used to start backprop)
print(v_res)  # tensor(12., grad_fn=<SumBackward0>)

# Print the gradient of v1
# ∂v_res/∂v1 = 2 because v_res = sum((v1 + v2) * 2)
# d/dv1 of that expression = 2 for each element
print(v1.grad)  # tensor([2., 2.])

# v2 doesn’t require grad, so PyTorch did not track it
# Therefore, v2.grad will be None
print(v2.grad)  # None
