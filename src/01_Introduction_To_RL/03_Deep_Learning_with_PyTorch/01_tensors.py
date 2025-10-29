import numpy as np
import torch

# Create an uninitialized (random) FloatTensor with shape (3, 2)
a = torch.FloatTensor(3, 2)
print(a)

# Create a tensor of all zeros with shape (3, 4)
zeros = torch.zeros(3, 4)
print(zeros)

# In-place operation: set all elements of 'a' to zero
print(a.zero_())

# Create a FloatTensor directly from a nested Python list
tensor_iterable_setup = torch.FloatTensor([[1, 2, 3], [3, 2, 1]])
print("\n", tensor_iterable_setup)

print("-------------------")

# Create a NumPy array of zeros with shape (3, 2), default dtype=float64
n = np.zeros(shape=(3, 2))
print(n)
print(n.dtype)  # Show the data type (float64 by default)

# Convert the NumPy array 'n' into a PyTorch tensor
b = torch.tensor(n)
print(b)
print(b.dtype)  # PyTorch infers dtype=torch.float64 since NumPy array was float64

print("\n-------------------")

# Create a NumPy array of zeros explicitly as 32-bit floats
n_32_bit = np.zeros(shape=(3, 2), dtype=np.float32)
print(n_32_bit)
print(n_32_bit.dtype)  # Confirm float32 dtype in NumPy

# Convert the float32 NumPy array to a PyTorch tensor (preserves dtype)
print(torch.tensor(n_32_bit))
print(torch.tensor(n_32_bit).dtype)  # dtype will be torch.float32

# Explicitly specify the dtype when converting NumPy -> PyTorch tensor
c = np.zeros(shape=(3, 2))
result = torch.tensor(c, dtype=torch.float32)
print(result)
print(result.dtype)  # dtype is torch.float32 due to explicit cast
