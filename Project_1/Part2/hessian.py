import torch
from functorch import hessian
from torch.nn.utils import _stateless
import time

# Create model
model = torch.nn.Sequential(torch.nn.Linear(1, 100), torch.nn.Tanh(), torch.nn.Linear(100, 1))
num_param = sum(p.numel() for p in model.parameters())
names = list(n for n, _ in model.named_parameters())

# Create random dataset
x = torch.rand((1000,1))
y = torch.rand((1000,1))

# Define loss function
def loss(params):
    y_hat = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x)
    return ((y_hat - y)**2).mean()

# Calculate Hessian
hessian_func = hessian(loss)

start = time.time()

what = tuple(model.parameters())

H = hessian_func(tuple(model.parameters()))
print(type(H))
H = torch.cat([torch.cat([e.flatten() for e in Hpart]) for Hpart in H]) # flatten
print(type(H))
print(H.size())
H = H.reshape(num_param, num_param)
print(type(H))
print(H.size())

print(H)

print(time.time() - start)