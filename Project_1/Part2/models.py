import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import os
from sklearn.decomposition import PCA
from csv import writer

class DNN_0(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 2 hidden layers, 608 parameters
        self.lin = nn.Linear(1, 34).to(device)
        self.fc1 = nn.Linear(34, 14).to(device)
        self.fc2 = nn.Linear(14, 7).to(device)
        self.lout = nn.Linear(7, 1).to(device)
        
        self.name = 'DNN_0'
        
    def forward(self, x):
        activation_func = F.relu
        x = activation_func(self.lin(x))
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.lout(x))
        x = self.lout(x)
        
        return x
    
class DNN_1(nn.Module):
    def __init__(self):
        super(DNN_1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1*28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_1'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)

def train_model_weight_pca(model, trainloader, testloader, epochs, 
                optimizer, loss_fn, device):
    
    # Create PCA object for dimension reduction
    pca = PCA(n_components=2)
    first_layer_weights = []
    
    optimizer = optimizer(model.parameters())
    
    csv_name = 'model_data/' + model.name + '_pca.csv'
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(trainloader):
            
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Get dimensionally reduced weight values
        if epoch % 3 == 0:
            # Get the weight tensor
            for name, param in model.named_parameters():
                first_layer_weights.append(param.view(-1))
        
    # Convert list of tensors to one tensor
    first_layer_weights = torch.stack(first_layer_weights)
    
    pca_first_layer_weights = pca.fit_transform(first_layer_weights)
    
    # Write the data to a .csv file
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['epochs', 'pca_1', 'pca_2'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(pca_first_layer_weights)
        
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')

    
def train_model_grad_norm_exp1(model, trainloader, testloader, epochs, 
                optimizer, loss_fn, device):
    
    optimizer = optimizer(model.parameters())
    training_info = []
    
    csv_name = 'model_data/' + model.name + '_grad_vs_loss.csv'
    training_running_loss = 0.0
    training_running_grad_norm = 0.0
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(trainloader):
            
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate grad norm
            grad_norm = get_grad_norm(model)
            
            training_running_loss += loss
            training_running_grad_norm += grad_norm
        
        # Caluclate average loss and grad norm for the epoch
        training_running_loss = round(training_running_loss.detach().cpu().item(), 3)
        training_running_grad_norm = round(training_running_grad_norm.detach().cpu().item(), 3)  
        average_grad_norm = training_running_grad_norm / len(trainloader)
        average_loss = training_running_loss / len(trainloader)
        training_info.append([epoch, average_loss, average_grad_norm])
        
    # Write the data to a .csv file
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['epochs', 'loss', 'grad_norm'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(training_info)
        
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
            
def train_model_grad_norm_exp2(model, trainloader, epochs,
                               optimizer, criterion, device):
    pass

def get_grad_norm(model):
    
    grad_all = 0.0
    
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
        
    return grad_all ** 0.5
            
            