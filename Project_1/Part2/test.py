import torch
from torch import nn, optim
import numpy as np

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        
        return x
    
model = Model()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
for param_tensor in model.state_dict():
    print(model.state_dict()[param_tensor])
    
with torch.no_grad():
    print(model.fc1.weight)
    model.fc1.weight[:,:] = torch.zeros(10, 10)
    print(model.fc1.weight)
    
arr = np.zeros([10,10])
tens = torch.from_numpy(arr)
for param_tensor in model.state_dict():
    model.state_dict()[param_tensor] = tens
    print(model.state_dict()[param_tensor])
    break

for param_tensor in model.state_dict():
    print(model.state_dict()[param_tensor])

print()