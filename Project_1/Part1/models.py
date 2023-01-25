import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv

class network1(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 2 hidden layers, 608 parameters
        self.lin = nn.Linear(1, 34).to(device)
        self.fc1 = nn.Linear(34, 14).to(device)
        self.fc2 = nn.Linear(14, 7).to(device)
        self.lout = nn.Linear(7, 1).to(device)
        
    def forward(self, x):
        activation_func = F.tanh
        x = activation_func(self.lin(x))
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = self.lout(x)
        
        return x

class network2(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 4 hidden layers, 607 parameters
        self.lin = nn.Linear(1, 10).to(device)
        self.fc1 = nn.Linear(10, 10).to(device)
        self.fc2 = nn.Linear(10, 10).to(device)
        self.fc3 = nn.Linear(10, 10).to(device)
        self.fc4 = nn.Linear(10, 27).to(device)
        self.lout = nn.Linear(27, 1).to(device)
    
    def forward(self, x):
        activation_func = F.relu
        x = activation_func(self.lin(x))
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = activation_func(self.fc4(x))
        x = self.lout(x)
        
        return x
    
class network3(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 2 hidden layers, 608 parameters
        self.lin = nn.Linear(1, 30).to(device)
        self.fc1 = nn.Linear(30, 17).to(device)
        self.fc2 = nn.Linear(17, 8).to(device)
        self.lout = nn.Linear(8, 1).to(device)
        
    def forward(self, x):
        activation_func = F.tanh
        x = activation_func(self.lin(x))
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = self.lout(x)
        
        return x
        
    
def train_model(model, x_train, y_train, x_eval, epochs, csv_name, convergence=.001):
    
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss(reduction='mean')
    loss_list = [['epochs', 'loss']]
    for epoch in range(epochs):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss}')
            loss_list.append([epoch, loss])
        
        # compute gradients
        loss.backward()
        
        # carry out one optimization step with Adam
        optimizer.step()
        
        # reset gradients to zero
        optimizer.zero_grad()
        
        if loss < convergence:
            print(f'Epoch: {epoch}, Loss: {loss}')
            print(f'Model converges after {epoch} epochs.')
            loss_list.append([epoch, loss])
            break
    
    print("-----------------------------------------")
    
    for index, (epoch, loss) in enumerate(loss_list):
        if index == 0:
            continue
        # # loss = torch.Tensor.cpu(loss)
        loss_list[index][1] = loss.item()
        # loss_list[index][1] = loss.detach().cpu().numpy()[0]
        
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loss_list)
    
    pred_eval = model(x_eval)
    
    # delete the labels from loss_list list and convert to dtype float
    loss_list = np.array(loss_list)
    loss_list = loss_list[1:]
    loss_list = loss_list.astype(np.float32)
    
    return pred_eval.detach().cpu().numpy(), loss_list


        
        