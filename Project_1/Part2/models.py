import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
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
        x = self.lout(x)
        
        return x
    
class DNN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_1'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
def create_model(model_type: str, checkpoint: str = None):
    model = None
    if model_type is None:
        print('Please enter a model type argument')
        exit()
  
    elif model_type == 'dnn_0':
        model = DNN_0()
    elif model_type == 'dnn_1':
        model = DNN_1()

    if checkpoint:
        model = torch.load(checkpoint)
        
    return model
    
def train_model_pca(model, training_dataloader, testing_dataloader, epochs, optimizer, loss_fn, device, count):
    
    first_layer_weights = []
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
    training_info = []
    
    csv_name = f'model_data/{model.name}_{count}.csv'
    
    training_running_loss = 0.0
    testing_running_loss = 0.0
    batch_size = 0
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////')
        
    for epoch in range(epochs):
        train_count = 0
        test_count = 0
        for batch, (img, label) in enumerate(training_dataloader):
            
            batch_size = len(img)
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_running_loss += loss
            train_count += 1
            if batch % 100 == 0:
                test_total_examples, testing_accuracy, testing_loss = test_accuracy(model, testing_dataloader, loss_fn, device)
                testing_running_loss += testing_loss
                test_count += 1
        print(f'Epoch {epoch + 1} completed')
        if epoch % 3 == 0:
            for name, param in model.named_parameters():
                print(param.view(-1))
                first_layer_weights.append(param.view(-1).detach().cpu().numpy().flatten())
                break
            
        total_epochs = model.training_epochs+epoch+1
        # training_running_loss = round(training_running_loss.detach().cpu().item(), 3)
        # testing_running_loss = round(testing_running_loss.detach().cpu().item(), 3)
        # train_total_examples, training_accuracy, training_loss = test_accuracy(model, training_dataloader, loss_fn, device)
        # test_total_examples, testing_accuracy, testing_loss = test_accuracy(model, testing_dataloader, loss_fn, device)
        # average_train_loss = training_running_loss/(train_count)
        # average_test_loss = testing_running_loss/(test_count)
        # print(f'Total Epochs: {total_epochs}, Training Ex Per Epoch: {train_total_examples}')
        # print(f'Average Training Loss: {average_train_loss}, Training Set Accuracy: {training_accuracy}')
        # print(f'Average Testing Loss: {average_test_loss}, Testing Accuracy ({test_total_examples} images): {testing_accuracy}')
        # print('-----------------------------------------------------------------------------')
        # training_info.append([total_epochs, average_train_loss, training_accuracy, average_test_loss, testing_accuracy])
        # training_running_loss = 0.0
        # testing_running_loss = 0.0
        
    # PCA analysis
    
    # convert the first layer weight list into proper data structure for pca
    first_layer_weights = np.array(first_layer_weights)
    
    # Scale the weight data
    scaling = preprocessing.StandardScaler()
    
    scaling.fit(first_layer_weights)
    scaled_weights = scaling.transform(first_layer_weights)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_weights)
    
    first_layer_pca = pca.transform(scaled_weights)
    
    print(first_layer_pca)
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    
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
        writer_object.writerows(first_layer_pca)
    
    model.training_epochs += epochs
    
def test_accuracy(model, dataloader, loss_fn, device):
    total = 0
    correct = 0
    
    with torch.no_grad():
        for  batch, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_fn(pred, label)
            
            _, predictions = torch.max(pred, 1)
            
            total += label.size(0)
            correct += (predictions == label).sum().item()
            
    return total, correct/total, loss
    
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
    
    optimizer = optimizer(model.parameters())
    
    
    
    '''Train on the initial loss function for 4 epochs, then switch loss
    function to gradient norm'''
    
    for epochs in range(4):
        
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            

def get_grad_norm(model):
    
    grad_all = 0.0
    
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = torch.sum((p.grad.cpu().data.numpy() ** 2))
        grad_all += grad
        
    return grad_all ** 0.5
            
            
