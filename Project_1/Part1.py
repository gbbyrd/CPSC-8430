'''
Grayson Byrd
CPSC 8430 - Homework 1, Part 1

Desc:
This is the file for part 1-1 of Homwork 1. We will train two different
DNN models with the same amount of parameters to simulate a chosen function.
Teh training process of each of the models will be compared using various
methods and the predictions of each model will be visualized in a graph
using matplotlib.
'''

import numpy as np
import pandas as pd
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import create_datasets
import argparse

# For debugging
parser = argparse.ArgumentParser(description="Determine if debugging.")
parser.add_argument('--debugging', action='store_true', required=False)
args = parser.parse_args()

# Set device hardware for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_folder = 'data/'
training_dataset = os.path.join(data_folder, 'train_dataset_1_1.csv')
testing_dataset = os.path.join(data_folder, 'test_dataset_1_1.csv')
path_to_current_folder = ""
if args.debugging:
    path_to_current_folder = "Project_1"
data_folder_path = os.path.join(path_to_current_folder, data_folder)
    
# dataset parameters
num_training_data_points = 4000
num_testing_data_points = 1000

# Create 3 deep neural network classes to train and compare
# Each network will have roughly 614 (arbitrarily chosen number)
# neurons/parameters

class network1(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 hidden layers, 608 parameters
        self.lin = nn.Linear(1, 34)
        self.fc1 = nn.Linear(34, 14)
        self.fc2 = nn.Linear(14, 7)
        self.fc3 = nn.Linear(7, 1)
        
        # set the dropout rate (to help prevent overfitting)
        # in this case, it is impossible to overfit, so we will leave this out
        
    def forward(self, x):
        # using relu here
        # activation_func = F.relu
        x = (self.lin(x))
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    # Create functions for the training loop and testing of the model
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        print(torch.numel(pred))
        print(pred)
        print(torch.numel(y))
        print(y)
        loss = loss_fn(pred, y)
        
        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss: >77f} [{current:>5d}/{size:>5d}")
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if pred == y:
                correct += 1
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

# Create a custom dataset class for the generated dataset
class RealFunctionDataset(Dataset):
    '''Real Function Dataset'''
    
    def __init__(self, csv_file, root_dir, transform=None):
        
        self.data_frame = pd.read_csv(os.path.join(root_dir, csv_file), dtype=np.float32)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print('idx ({idx}) was a tensor')
        
        input = torch.tensor([self.data_frame.iloc[idx, 0]])
        output = torch.tensor([self.data_frame.iloc[idx, 1]])
        # if self.transform:
        #     input = self.transform(input)
        # if self.transform:
        #     output = self.transform(output)
        
        return input, output

if __name__=='__main__':
    
    # The real function to simulate: (e**sin(x))*cos(x)

    # Create the training dataset if not already created
    '''
    The data sets will be constrained from -10 to 10. 4000 data points will 
    be randomly generated for the training set and 1000 for the testing set.
    The numbers will have up to 3 decimal places of accuracy.
    '''
    if not os.path.exists(data_folder_path):
        # Create the training dataset
        create_datasets.create_part1_dataset(training_dataset, num_training_data_points)
        
        # Create the testing dataset
        create_datasets.create_part1_dataset(testing_dataset, num_testing_data_points)
    
    # Instantiate the custom datasets
    training_dataset = RealFunctionDataset(csv_file=training_dataset,
                                                root_dir=path_to_current_folder)
    testing_dataset = RealFunctionDataset(csv_file=testing_dataset,
                                                root_dir=path_to_current_folder)
    
    # Load into the dataloader
    train_dataloader = DataLoader(training_dataset, batch_size = 64, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True)
    
    # Check dataloader batch size
    train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    print (train_features)
    print(train_labels)
    
    # Instantiate model
    model1 = network1()
    
    # Set hyperparameters for training
    '''
    Hyperparameters for training include:
    - Number of Epochs
    - Batch Size
    - Learning Rate
    '''
    
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    
    # Optimization loop
    
    # Choose a loss function and optimizer
    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------------")
        train_loop(train_dataloader, model1, loss_fn, optimizer)
        test_loop(test_dataloader, model1, loss_fn)
    print("Done!")
    
    
    
    
    
    
    
    

        
    
        
    
    
    
    
