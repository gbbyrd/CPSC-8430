'''
Grayson Byrd
CPSC 8430 - Deep Learning
Part 3 - Number of Parameters vs Generalization

The purpose of this part of the assignment is to display the effect that
the number of parameters of a model has on its ability to generalize. Is
more parameters always better?

We will test this using 10 models training on the MNIST dataset.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from csv import writer

import models
import args

arguments = args.parser.parse_args()

checkpoint_folder_path = 'checkpoints/'
    
# Set GPU device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                          shuffle=True, num_workers=2)
trainloader_validation = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                       shuffle=True, num_workers=2)
testloader_validation = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)

model_name_list = [
    'dnn_0',
    'dnn_1',
    'dnn_2',
    'dnn_3',
    'dnn_4',
    'dnn_5',
    'dnn_6',
    'dnn_7',
    'dnn_8',
    'dnn_9'
]

def run_model(model_name, checkpoint_path=None):
    optimizer = optim.Adam
    loss_fn = nn.CrossEntropyLoss()
    
    csv_name = 'model_data/num_params_analysis.csv'
    
    model = models.create_model(model_name, checkpoint_path)
    model = model.to(DEVICE)
    models.train_model(model, trainloader, trainloader_validation, testloader_validation, arguments.epochs,
                       optimizer, loss_fn, DEVICE)
    testing_total, testing_accuracy, testing_loss = models.test_accuracy(model, testloader_validation, 
                                                                              loss_fn, DEVICE)
    training_total, training_accuracy, training_loss = models.test_accuracy(model, trainloader_validation,
                                                                            loss_fn, DEVICE)
    testing_loss = round(testing_loss.detach().cpu().item(), 3)
    training_loss = round(training_loss.detach().cpu().item(), 3)
    num_parameters = models.count_params(model)
    
    validation_info = [num_parameters, testing_accuracy, training_accuracy, testing_loss, training_loss]
    
     # Create model_data directory
    if not os.path.exists('model_data/'):
        os.mkdir('model_data/')
    
    # Write training data to csv file
    if not os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            writer_object = writer(f)
            writer_object.writerow(['num_params', 'testing_accuracy', 'training_accuracy', 'testing_loss', 'training_loss'])
            
            f.close()
    
    with open(csv_name, 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(validation_info)
    
    
def main():
    for name in model_name_list:
        run_model(name)
    
if __name__ == '__main__':
    main()


