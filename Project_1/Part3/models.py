import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from csv import writer

class CNN_0(nn.Module):
    def __init__(self):
        super(CNN_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=3)
        self.batch1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=3)
        self.conv3 = nn.Conv2d(64, 10, 5)
        self.batch2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(40, 145)
        self.fc2 = nn.Linear(145, 30)
        self.fc3 = nn.Linear(30, 10)
        
        # Keep track of training epochs for loading and continuing
        # model training
        self.training_epochs = 0
        self.name = 'cnn_0'
        
    def forward(self, x):
        # Conv Layer 1
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        
        # Pooling Layer 1
        x = self.pool(x)
        
        # Droupout
        x = self.dropout(x)
        
        # Conv Layer 2 
        x = self.conv2(x)
        x = F.relu(self.batch1(x))
        
        # Pooling Layer 2
        x = self.pool(x)
        
        # Droupout
        x = self.dropout(x)
        
        # Conv Layer 3
        x = self.conv3(x)
        x = F.relu(self.batch2(x))
        
        # Pooling layer 3
        x = self.pool(x)
        
        # Droupout
        x = self.dropout(x)
        
        # Flatten tensor for FCLs
        x = torch.flatten(x, 1)
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class CNN_1(nn.Module):
    def __init__(self) -> None:
        super(CNN_1, self).__init__()
        self.conv_and_pool = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        self.training_epochs = 0
        self.name = 'cnn_1'
        
    def forward(self, x):
        x = self.conv_and_pool(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected(x)
        
        return x

# Adds dropout to the cnn_2 model
class CNN_2(nn.Module):
    def __init__(self) -> None:
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.batch1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(.25)
        
        self.training_epochs = 0
        self.name = 'cnn_2'
        
    def forward(self, x):
        act_func = F.relu
        x = self.conv1(x)
        x = act_func(self.batch1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = act_func(self.batch2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = act_func(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.training_epochs = 0
        self.name = 'cnn_3'
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output
    
class CNN_4(nn.Module):
    def __init__(self):
        super(CNN_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.training_epochs = 0
        self.name = 'cnn_4'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# Base model to improve upon
class DNN_0(nn.Module):
    def __init__(self) -> None:
        super(DNN_0, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 55)
        self.fc4 = nn.Linear(55, 10)
        
        # Keep track of training epochs for loading and continuing
        # model training
        self.training_epochs = 0
        self.name = 'dnn_0'
        
    def forward(self, x):
        activation_func = F.relu
        # Flatten the rgb data
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = activation_func(self.fc2(x))
        x = activation_func(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
class DNN_1(nn.Module):
    
    def __init__(self) -> None:
        super(DNN_1, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 400)
        self.fc2 = nn.Linear(400, 500)
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, 700)
        self.fc5 = nn.Linear(700, 400)
        self.fc6 = nn.Linear(400, 50)
        self.fc7 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(.25)
        
        self.training_epochs = 0
        self.name = 'dnn_1'
        
    def forward(self, x):
        activation_func = F.relu
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = self.dropout(x)
        x = activation_func(self.fc2(x))
        x = self.dropout(x)
        x = activation_func(self.fc3(x))
        x = self.dropout(x)
        x = activation_func(self.fc4(x))
        x = self.dropout(x)
        x = activation_func(self.fc5(x))
        x = self.dropout(x)
        x = activation_func(self.fc6(x))
        x = self.dropout(x)
        x = self.fc7(x)
        
        return x
        
class DNN_2(nn.Module):
    def __init__(self) -> None:
        super(DNN_2, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 100)
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
        self.name = 'dnn_2'
        
    def forward(self, x):
        activation_func = F.relu
        x = torch.flatten(x, 1)
        x = activation_func(self.fc1(x))
        x = self.dropout(x)
        x = activation_func(self.fc2(x))
        x = self.dropout(x)
        x = activation_func(self.fc3(x))
        x = self.dropout(x)
        x = activation_func(self.fc4(x))
        x = self.dropout(x)
        x = activation_func(self.fc5(x))
        x = self.dropout(x)
        x = activation_func(self.fc6(x))
        x = self.dropout(x)
        x = activation_func(self.fc7(x))
        x = self.dropout(x)
        x = activation_func(self.fc8(x))
        x = self.dropout(x)
        x = activation_func(self.fc9(x))
        x = self.dropout(x)
        x = self.fc10(x)
        
        return x
    
class DNN_3(nn.Module):
    def __init__(self):
        super(DNN_3, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1*28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 10),
        )
        
        self.training_epochs = 0
        self.name = 'dnn_3'
        
    def forward(self, x):
        return self.network(x)
    
class DNN_4(nn.Module):
    def __init__(self):
        super(DNN_4, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        self.training_epochs = 0
        self.name = 'dnn_4'
        
    def forward(self, x):
        return self.network(x)
    
def create_model(model_type: str, checkpoint: str):
    model = None
    if model_type is None:
        print('Please enter a model type argument')
        exit()
    elif model_type == 'cnn_0':
        model = CNN_0()
    elif model_type == 'cnn_1':
        model = CNN_1()
    elif model_type == 'cnn_2':
        model = CNN_2()
    elif model_type == 'cnn_3':
        model = CNN_3()
    elif model_type == 'cnn_4':
        model = CNN_4()
    elif model_type == 'dnn_0':
        model = DNN_0()
    elif model_type == 'dnn_1':
        model = DNN_1()
    elif model_type == 'dnn_2':
        model = DNN_2()
    elif model_type == 'dnn_3':
        model = DNN_3()
    elif model_type == 'dnn_4':
        model = DNN_4()
        
    if checkpoint:
        model = torch.load(checkpoint)
        
    return model

def save_model(model):
    if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints')
            
    PATH = 'checkpoints/' + str(model.name) + '_' + str(model.training_epochs) + '.pth'
    torch.save(model, PATH)
    
    return
    
def train_model(model, training_dataloader, testing_dataloader, epochs, optimizer, loss_fn, device):
    
    optimizer = optimizer(model.parameters())
    loss_fn = loss_fn
    
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
    