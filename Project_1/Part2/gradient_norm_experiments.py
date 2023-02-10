import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import models
from torch import optim

import generate_figures

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
    
# Create custom dataset for chosen function
class FunctionDataset(Dataset):
    def __init__(self, function, test=False):
        self.x = np.linspace(-5, 5, num=200).reshape(-1, 1)
        self.x = torch.from_numpy(self.x).float()
        self.y = self.test_functions(self.x, function)
        if test:
            self.x = np.linspace(-5, 5, num=137).reshape(-1, 1)
            self.x = torch.from_numpy(self.x).float()
            self.y = self.test_functions(self.x, function)
            
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return len(self.x)
    
    def test_functions(self, x, function: str):
        funcs = {
            'cosine': np.cos(x), # cosine function
            'sine': np.sin(x), # sine function
            'exponential': np.exp(x), # exponential function
            'power': np.power(x, 3), # cubic function
            'quadratic': np.power(x, 2), # quadratic
            'crazy_sin': np.sin(5*np.pi*x)/(5 * np.pi * x)
        }
        return funcs.get(function)
    
def main():
    # Choose function
    chosen_function = 'sine'
    
    # Create dataset and dataloader
    trainset = FunctionDataset(chosen_function)
    testset = FunctionDataset(chosen_function, test=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    ''' Experiment 1 '''
    # Create model
    model_exp1 = models.DNN_0(device)
    
    # Define optimizer and criterion
    optimizer = optim.Adam
    criterion = nn.MSELoss(reduction='mean')
    exp_1_epochs = 300
    
    # Trains the model and records the loss and gradient norm for comparison
    models.train_model_grad_norm_exp1(model_exp1, trainloader, testloader, exp_1_epochs,
                       optimizer, criterion, device)
    
    generate_figures.plot_grad_vs_loss()
    
    ''' Experiment 2 '''
    # Create model 
    model_exp2 = models.DNN_0(device)
    
    # Define optimizer and criterion
    exp_2_epochs = 100
    
    hessian_matrix = models.train_model_grad_norm_exp2(model_exp2, trainloader, exp_2_epochs, 
                                                       optimizer, criterion, device)
    
    generate_figures.plot_min_figures()
    
if __name__ == '__main__':
    main()