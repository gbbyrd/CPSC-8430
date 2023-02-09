import torch
import torch.nn as nn
import os
from csv import writer
import torch.nn.functional as F
import numpy as np

class DNN_Random_Fit(nn.Module):
    def __init__(self):
        super(DNN_Random_Fit, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_random_fit'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_0(nn.Module):
    def __init__(self):
        super(DNN_0, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_0'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_1(nn.Module):
    def __init__(self):
        super(DNN_1, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 50),
            nn.ReLU(),
            nn.Linear(50, 300),
            nn.ReLU(),
            nn.Linear(300, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_1'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_2(nn.Module):
    def __init__(self):
        super(DNN_2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_2'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_3(nn.Module):
    def __init__(self):
        super(DNN_3, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 200),
            nn.ReLU(),
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_3'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_4(nn.Module):
    def __init__(self):
        super(DNN_4, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_4'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_5(nn.Module):
    def __init__(self):
        super(DNN_5, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_5'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_6(nn.Module):
    def __init__(self):
        super(DNN_6, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_6'
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.network(x)
    
class DNN_7(nn.Module):
    def __init__(self) -> None:
        super(DNN_7, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 200)
        self.fc6 = nn.Linear(200, 300)
        self.fc7 = nn.Linear(300, 50)
        self.fc8 = nn.Linear(50, 75)
        self.fc9 = nn.Linear(75, 100)
        self.fc10 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(.25)
        
        self.training_epochs = 0
        self.name = 'dnn_7'
        
    def forward(self, x):
        activation_func = F.relu
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = activation_func(self.fc4(x))
        x = activation_func(self.fc5(x))
        x = activation_func(self.fc6(x))
        x = activation_func(self.fc7(x))
        x = activation_func(self.fc8(x))
        x = activation_func(self.fc9(x))
        x = self.fc10(x)
        
        return x
    
class DNN_8(nn.Module):
    def __init__(self) -> None:
        super(DNN_8, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 55)
        self.fc4 = nn.Linear(55, 10)
        
        # Keep track of training epochs for loading and continuing
        # model training
        self.training_epochs = 0
        self.name = 'dnn_8'
        
    def forward(self, x):
        activation_func = F.relu
        # Flatten the rgb data
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
class DNN_9(nn.Module):
    
    def __init__(self) -> None:
        super(DNN_9, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 400)
        self.fc2 = nn.Linear(400, 500)
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, 700)
        self.fc5 = nn.Linear(700, 400)
        self.fc6 = nn.Linear(400, 500)
        self.fc7 = nn.Linear(500, 50)
        self.fc8 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(.25)
        
        self.training_epochs = 0
        self.name = 'dnn_9'
        
    def forward(self, x):
        activation_func = F.relu
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = activation_func(self.fc4(x))
        x = activation_func(self.fc5(x))
        x = activation_func(self.fc6(x))
        x = activation_func(self.fc7(x))
        x = self.fc8(x)
        
        return x
    
class DNN_10(nn.Module):
    def __init__(self, name) -> None:
        super(DNN_10, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self.name = name
        self.training_epochs = 0
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
'''Model related functions:
Training, testing, creating models etc.'''

def count_params(model):
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    
    return total_params

def create_model(model_name: str, checkpoint: str = None):
    model = None
    if model_name is None:
        print('Please enter a model type argument')
        exit()
    elif model_name == 'dnn_0':
        model = DNN_0()
    elif model_name == 'dnn_1':
        model = DNN_1()
    elif model_name == 'dnn_2':
        model = DNN_2()
    elif model_name == 'dnn_3':
        model = DNN_3()
    elif model_name == 'dnn_4':
        model = DNN_4()
    elif model_name == 'dnn_5':
        model = DNN_5()
    elif model_name == 'dnn_6':
        model = DNN_6()
    elif model_name == 'dnn_7':
        model = DNN_7()
    elif model_name == 'dnn_8':
        model = DNN_8()
    elif model_name == 'dnn_9':
        model = DNN_9()
    elif model_name == 'dnn_random_fit':
        model = DNN_Random_Fit()
        
    if checkpoint:
        model = torch.load(checkpoint)
        
    return model

def save_model(model):
    if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints')
            
    PATH = 'checkpoints/' + str(model.name) + '_' + str(model.training_epochs) + '.pth'
    torch.save(model, PATH)
    
    return
    
def train_model(model, training_dataloader, training_dataloader_validation, testing_dataloader_validation, epochs, optimizer, loss_fn, device):
    
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
    training_info = []
    
    csv_name = 'model_data/' + model.name + '.csv'
    
    training_running_loss = 0.0
    testing_running_loss = 0.0
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////')
        
    for epoch in range(epochs):
        train_count = 0
        test_count = 0
        for batch, (img, label) in enumerate(training_dataloader):
            
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_running_loss += loss
            train_count += 1
            
        total_epochs = model.training_epochs+epoch+1
        train_total_examples, training_accuracy, training_loss = test_accuracy(model, training_dataloader_validation, loss_fn, device)
        test_total_examples, testing_accuracy, testing_loss = test_accuracy(model, testing_dataloader_validation, loss_fn, device)
        training_loss = training_loss.detach().cpu().item()
        testing_loss = testing_loss.detach().cpu().item()
        print(f'Total Epochs: {total_epochs}, Training Ex Per Epoch: {train_total_examples}')
        print(f'Average Training Loss: {training_loss}, Training Set Accuracy: {training_accuracy}')
        print(f'Average Testing Loss: {testing_loss}, Testing Accuracy ({test_total_examples} images): {testing_accuracy}')
        print('-----------------------------------------------------------------------------')
        training_info.append([total_epochs, training_loss, training_accuracy, testing_loss, testing_accuracy])
        training_running_loss = 0.0
        testing_running_loss = 0.0
        
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
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
    
    checkpoint_path = f'checkpoints/{model.name}.pth'
    torch.save(model, checkpoint_path)
    
def train_model_sensiti(model, training_dataloader, testing_dataloader, epochs, optimizer, loss_fn, device):
    
    num_parameters = count_params(model)
    
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
    training_info = []
    
    csv_name = 'model_data/' + model.name + '.csv'
    
    training_running_loss = 0.0
    testing_running_loss = 0.0
    
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////')
        
    for epoch in range(epochs):
        train_count = 0
        test_count = 0
        for batch, (img, label) in enumerate(training_dataloader):
            
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
        training_info.append([total_epochs, average_train_loss, training_accuracy, average_test_loss, testing_accuracy, num_parameters])
        training_running_loss = 0.0
        testing_running_loss = 0.0
        
    print('//////////////////////////////// TRAINING /////////////////////////////////////////////////////////////////////////////////////\n\n')
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['epochs', 'training_loss', 'training_accuracy', 'testing_loss', 'testing_accuracy', 'num_parameters'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(training_info)
    
    model.training_epochs += epochs
    
    checkpoint_path = f'checkpoints/{model.name}.pth'
    torch.save(model, checkpoint_path)
                
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

def linear_interpolation_of_two_models(model1, model2, alpha):
    ''' This function assumes both models are a one layer model with the input
    layer named fc1 and the output named fc2 '''
    
    # Get weight and bias tensors
    model1_weight1 = model1.fc1.weight
    model1_bias1 = model1.fc1.bias
    model1_weight2 = model1.fc2.weight
    model1_bias2 = model1.fc2.bias
    
    model2_weight1 = model2.fc1.weight
    model2_bias1 = model2.fc1.bias
    model2_weight2 = model2.fc2.weight
    model2_bias2 = model2.fc2.bias
    
    # Create new model
    new_model = DNN_10('new_model')
    
    
    # Get weights and biases for new model
    with torch.no_grad():
        new_model.fc1.weight[:,:] = (1 - alpha)*model1_weight1 + alpha * model2_weight1
        new_model.fc1.bias[:] = (1 - alpha)*model1_bias1 + alpha * model2_bias1
        new_model.fc2.weight[:,:] = (1 - alpha)*model1_weight2 + alpha * model2_weight2
        new_model.fc2.bias[:] = (1 - alpha)*model1_bias2 + alpha * model2_bias2
        
    return new_model

def analyze_interpolations(model1, model2, trainloader, testloader, criterion, device):
    x = np.linspace(-1, 2, num=30)
    csv_name = 'model_data/interpolated_model.csv'
    testing_info = []
    for alpha in x:
        new_model = linear_interpolation_of_two_models(model1, model2, alpha)
        new_model = new_model.to(device)
        test_total, testing_accuracy, test_loss = test_accuracy(new_model, testloader, criterion, device)
        train_total, train_accuracy, train_loss = test_accuracy(new_model, trainloader, criterion, device)
        test_loss = round(test_loss.detach().cpu().item(), 3)
        train_loss = round(train_loss.detach().cpu().item(), 3)
        testing_info.append([alpha, testing_accuracy, test_loss, train_accuracy, train_loss])
        
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['alpha', 'test_accuracy', 'test_loss', 'train_accuracy', 'train_loss'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(testing_info)
        
def analyze_sensitivity(model_list, batch_size_list, criterion, trainloader, testloader, device):
    analysis_info = []
    csv_name = 'model_data/sensitivity_analysis.csv'
    for idx, model in enumerate(model_list):
        model = model.to(device)
        sensitivity = get_grad_norm(model)
        sensitivity = sensitivity.detach().cpu().item()
        batch_size = batch_size_list[idx]
        testing_total, testing_accuracy, testing_loss = test_accuracy(model, testloader, criterion,
                                                                      device)
        training_total, training_accuracy, training_loss = test_accuracy(model, trainloader, criterion,
                                                                         device)
        testing_loss = testing_loss.detach().cpu().item()
        training_loss = training_loss.detach().cpu().item()
        analysis_info.append([batch_size, sensitivity, testing_accuracy, testing_loss, training_accuracy, training_loss])
        
    # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['batch_size', 'sensitivity', 'testing_accuracy', 'testing_loss', 'training_accuracy', 'training_loss'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerows(analysis_info)
        
def get_grad_norm(model):
    
    grad_all = 0.0
    
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = torch.sum((p.grad ** 2))
        grad_all += grad
        
    return grad_all ** 0.5