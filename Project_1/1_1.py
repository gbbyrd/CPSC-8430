'''
Grayson Byrd
CPSC 8430 - Homework 1

Desc:
This is the file for part 1-1 of Homwork 1. We will train two different
DNN models with the same amount of parameters to simulate a chosen function.
Teh training process of each of the models will be compared using various
methods and the predictions of each model will be visualized in a graph
using matplotlib.
'''

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import create_datasets

# Create 3 deep neural network classes to train and compare
# Each network will have roughly 614 (arbitrarily chosen number)
# neurons/parameters

class network1(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 hidden layers, 608 parameters
        self.fc1 = nn.Linear(1, 34)
        self.fc2 = nn.Linear(34, 14)
        self.fc3 = nn.Linear(14, 7)
        self.fc4 = nn.Linear(7, 1)
        
        # set the dropout rate (to help prevent overfitting)
        # in this case, it is impossible to overfit, so we will leave this out
        
    def forward(self, x):
        # using relu here
        activation_func = F.relu()
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    

if __name__=='__main__':
    
    # The real function to simulate: (e**sin(x))*cos(x)

    # Create the dataset
    '''
    The data set will be constrained from -10 to 10. 4000 data points will be randomly generated.
    The numbers will have up to 3 decimal places of accuracy.
    '''
    create_datasets.create_1_1_dataset()
    
    # Read in the dataset
    data = pd.read_csv('./dataset_1_1.csv', dtype=np.float32)
    
    data = np.array(data)
