import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import models
from torch import optim

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def test_functions(x, chosen_function):
    funcs = {
        'cosine': np.cos(x), # cosine function
        'sine': np.sin(x), # sine function
        'exponential': np.exp(x), # exponential function
        'power': np.power(x, 3), # cubic function
        'quadratic': np.power(x, 2), # quadratic
        'crazy_sin': np.sin(5*np.pi*x)/(5 * np.pi * x)
    }
    return funcs.get(chosen_function)
    
# Create custom dataset for chosen function
class FunctionDataset(Dataset):
    def __init__(self, function, test=False):
        self.x = np.linspace(-5, 5, num=200).reshape(-1, 1)
        self.y = function(self.x)
        if test:
            self.x = np.linspace(-5, 5, num=137).reshape(-1, 1)
            self.y = function(self.x)
            
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return len(self.x)
    
def main():
    # Choose function
    chosen_function = test_functions('')
    
    # Create dataset and dataloader
    trainset = FunctionDataset(chosen_function)
    testset = FunctionDataset(chosen_function, test=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # Create model
    model = models.DNN_0()
    
    # Define optimizer and criterion
    optimizer = optim.Adam()
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    
    models.train_model_grad_norm(model, trainloader, testloader, epochs,
                       optimizer, criterion, device)
    
if __name__ == '__main__':
    main()