'''
Grayson Byrd

Part 2 - Visualize the optimization process.

This code uses PCA and the first layer weights of the network to depict
the optimization process. To run this file, use the following command:

python3 visualize_optimization_process.py --train --test --epochs 100 --model_type dnn_1 --batch_size 32
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import models
import args
import os
import glob

arguments = args.parser.parse_args()

checkpoint_folder_path = 'checkpoints/'
batch_size = arguments.batch_size
checkpoint = arguments.checkpoint
model_type = arguments.model_type
epochs = arguments.epochs
train = arguments.train
test = arguments.test
is_all = arguments.all_models

if checkpoint is None:
    checkpoint_path = None
else:
    checkpoint_path = os.path.join(checkpoint_folder_path, checkpoint)

# Set GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def run_model(model_type, checkpoint_path, count):
    
    model = models.create_model(model_type, checkpoint_path)
    model = model.to(device)
    
    # Train the Models
    if train:
        
        optimizer = optim.Adam
        loss_fn = nn.CrossEntropyLoss()
        
        models.train_model_pca(model, trainloader, testloader, epochs, optimizer, loss_fn, device, count)
    
    ############ TESTING AND VALIDATION #######################
    
    if test:
    
        loss_fn = nn.CrossEntropyLoss()
        
        total_testing_examples, testing_accuracy, testing_loss = models.test_accuracy(model, testloader, loss_fn, device)

        print('//////////////////////////////// TESTING /////////////////////////////////////////////////////////////////////////////////////')
        print(f'{model.name} achieved {testing_accuracy} accuracy on {total_testing_examples} after {model.training_epochs} training epochs.')
        print('//////////////////////////////// TESTING /////////////////////////////////////////////////////////////////////////////////////\n\n')

def main():
    for i in range(8):    
        run_model(model_type, checkpoint_path)
        
    models.pca_analysis('/model_data/dnn_1.csv')
    
        
if __name__=='__main__':
    main()
