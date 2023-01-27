import torch
import torch.nn as nn
import torch.nn.functional as F
from csv import writer
import os
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.batch1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 10, 5)
        self.batch2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10*5*5, 145)
        self.fc2 = nn.Linear(145, 30)
        self.fc3 = nn.Linear(30, 10)
        
        # Keep track of training epochs for loading and continuing
        # model training
        self.training_epochs = 0
        
    def forward(self, x):
        # Conv Layer 1
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        
        # Pooling Layer 1
        x = self.pool(x)
        
        # Conv Layer 2 
        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        
        # Pooling Layer 2
        x = self.pool(x)
        
        # Flatten tensor for FCLs
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
class DNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 55)
        self.fc4 = nn.Linear(55, 10)
        
        # Keep track of training epochs for loading and continuing
        # model training
        self.training_epochs = 0
        
    def forward(self, x):
        activation_func = F.relu
        # Flatten the rgb data
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
def train_model(model, csv_name, dataloader, epochs, optimizer, loss_fn, device):
    
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
    training_info = []
    
    running_loss = 0.0
    for epoch in range(epochs):
        
        for batch, (img, label) in enumerate(dataloader):
            
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            
        total_epochs = model.training_epochs+epoch+1
        running_loss = running_loss.detach().cpu().item()
        print(f'Total Epochs: {total_epochs}, Average Loss: {running_loss/len(dataloader)}')
        training_info.append([total_epochs, round(running_loss/len(dataloader), 3)])
        running_loss = 0.0
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['epochs', 'average_loss'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(training_info)
                
def test_accuracy(model, dataloader, batch_size, device):
    total = 0
    correct = 0
    
    with torch.no_grad():
        for  batch, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            
            _, predictions = torch.max(pred, 1)
            
            total += label.size(0)
            correct += (predictions == label).sum().item()
            
    print(f'Accuracty after {total} test images: {correct / total}.')