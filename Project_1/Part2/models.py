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
import autograd_lib
from functorch import hessian
from torch.nn.utils import _stateless
import time


class DNN_0(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 2 hidden layers, 608 parameters
        self.lin = nn.Linear(1, 10).to(device)
        self.lout = nn.Linear(10, 1).to(device)
        
        self.name = 'DNN_0'
        
    def forward(self, x):
        activation_func = F.relu
        x = activation_func(self.lin(x))
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

def train_model(model, training_dataloader, testing_dataloader, epochs, optimizer, loss_fn, device):
    
    first_layer_weights = []
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    batch_size = ...
    training_info = []
    
    csv_name = 'model_data/' + model.name + '.csv'
    
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
            
        total_epochs = model.training_epochs+epoch+1
        training_running_loss = round(training_running_loss.detach().cpu().item(), 3)
        testing_running_loss = round(testing_running_loss.detach().cpu().item(), 3)
        train_total_examples, training_accuracy, training_loss = test_accuracy(model, training_dataloader, loss_fn, device)
        test_total_examples, testing_accuracy, testing_loss = test_accuracy(model, testing_dataloader, loss_fn, device)
        average_train_loss = training_running_loss/(train_count)
        average_test_loss = testing_running_loss/(test_count)
        print(f'Total Epochs: {total_epochs}, Training Ex Per Epoch: {train_total_examples}')
        print(f'Average Training Loss: {average_train_loss}, Training Set Accuracy: {training_accuracy}')
        print(f'Average Testing Loss: {average_test_loss}, Testing Accuracy ({test_total_examples} images): {testing_accuracy}')
        print('-----------------------------------------------------------------------------')
        training_info.append([total_epochs, average_train_loss, training_accuracy, average_test_loss, testing_accuracy])
        training_running_loss = 0.0
        testing_running_loss = 0.0
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['epochs', 'training_loss', 'training_accuracy', 'testing_loss', 'testing_accuracy'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(training_info)
    
    model.training_epochs += epochs
    
    checkpoint_path = f'checkpoints/{model.name}_{batch_size}.pth'
    torch.save(model, checkpoint_path)
    
    
    
def train_model_pca(model, training_dataloader, testing_dataloader, epochs, optimizer, loss_fn, device):
    
    first_layer_weights = []
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
    training_info = []
    
    csv_name = f'model_data/{model.name}.csv'
    
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
        
    # PCA analysis
    
    # convert the first layer weight list into proper data structure for pca
    first_layer_weights = np.array(first_layer_weights)
    print(first_layer_weights.shape)
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    # Write training data to csv file
    # if not os.path.exists(csv_name):
    #     with open(csv_name, 'a') as f:
    #         writer_object = writer(f)
    #         writer_object.writerow(['first_layer_weights'])
            
    #         f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(first_layer_weights)
        
        f.close()
    
    model.training_epochs += epochs
    
def pca_analysis(csv_path):
    
    # Get data
    first_layer_weights = pd.read_csv(csv_path, header=None)
    print(first_layer_weights.shape)
    
    # Scale the weight data
    scaling = preprocessing.StandardScaler()
    
    scaling.fit(first_layer_weights.values)
    scaled_weights = scaling.transform(first_layer_weights)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_weights)
    
    first_layer_pca = pca.transform(scaled_weights)
    
    print(first_layer_pca)
    
    pca_csv_name = 'model_data/pca_analysis.csv'
    
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    with open(pca_csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(first_layer_pca)
    
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
            grad_norm = grad_norm.cpu().data.numpy()
            
            training_running_loss += loss
            training_running_grad_norm += grad_norm
        
        # Caluclate average loss and grad norm for the epoch
        training_running_loss = round(training_running_loss.detach().cpu().item(), 3)
        training_running_grad_norm = round(training_running_grad_norm, 3)  
        average_grad_norm = training_running_grad_norm / len(trainloader)
        average_loss = training_running_loss / len(trainloader)
        training_info.append([epoch, average_loss, average_grad_norm])
        print(f'Epoch: {epoch}, Loss: {average_loss}')
        training_running_loss = 0
        training_running_grad_norm = 0
        
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
    
    for epoch in range(4):
        grad_norm = 0
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            grad_norm += get_grad_norm(model)
            
        average_grad_norm = grad_norm / len(trainloader)
        
        print(f'Epoch: {epoch}, Grad Norm: {average_grad_norm}')
        
    num_param = sum(p.numel() for p in model.parameters())
    names = list(n for n, _ in model.named_parameters())
    
    def loss(params):
        y_hat = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x)
        return ((y_hat - y)**2).mean()
    
    # Calculate the Hessian
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
    H = H.detach().cpu().numpy()
    eigenvalues = np.linalg.eig(H)
    
    print(eigenvalues)
    
    print(type(eigenvalues[0]))

    print(time.time() - start)
    
    # # Train until grad norm is approximately zero, then calculate the hessian matrix
    # hessian_matrix = ...
    # while (1):
    #     grad_norm = 0
    #     pred = 0
    #     for x, y in trainloader:
    #         x = x.to(device)
    #         y = y.to(device)
            
    #         pred = model(x)
    #         loss = get_grad_norm(model)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         grad_norm += get_grad_norm(model)
            
    #     average_grad_norm = grad_norm / len(trainloader)
        
    #     if average_grad_norm < .1:
    #         hessian_matrix = autograd_lib.backward_hessian(pred)
    #         break
        
    #     print(f'Epoch: {epoch}, Grad Norm: {average_grad_norm}')
        
    #     return hessian_matrix

def get_grad_norm(model):
    
    grad_all = 0.0
    
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = torch.sum((p.grad ** 2))
        grad_all += grad
        
    return grad_all ** 0.5
            
            
