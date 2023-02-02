'''
Grayson Byrd
CPSC 8430 - Deep Learning
Part 3 of Homework 1

The purpose of this part of the assignment is to demonstrate that a network
can learn randomized labels. This shows that the networks just fit the training
data, so in order to have high performing networks on data that it has not been
trained on, the training data must adequately represent the rest of the unseen data.

This will be done using the MNIST dataset using randomized labels.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import PIL
import numpy as np
import models
import args
import os
import generate_figures

arguments = args.parser.parse_args()

checkpoint_folder_path = 'checkpoints/'
checkpoint = arguments.checkpoint
train = arguments.train
test = arguments.test
epochs = arguments.epochs
batch_size = arguments.batch_size
model_type = arguments.model_type
is_all = arguments.all_models

if checkpoint is None:
    checkpoint_path = None
else:
    checkpoint_path = os.path.join(checkpoint_folder_path, checkpoint)
    
# Set GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Custom MNIST Wrapper for randomizing the labels of the dataset
class MyTwistedMNIST(torch.utils.data.Dataset):
    def __init__(self):
        super(MyTwistedMNIST, self).__init__()
        self.orig_mnist = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
        self.rand_label = np.random.randint(10, size=len(self.orig_mnist))
        
    def __getitem__(self, index):
        x, y = self.orig_mnist[index]
        
        return x, self.rand_label[index]
    
    def __len__(self):
        return self.orig_mnist.__len__()
    
# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset_twisted = MyTwistedMNIST()
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

trainloader_twisted = torch.utils.data.DataLoader(trainset_twisted, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# List of model class names to loop through to train all models at once
model_list = [
    'cnn_0',
    'cnn_1',
    'cnn_2',
    'cnn_3',
    'cnn_4',
    'dnn_0',
    'dnn_1',
    'dnn_2',
    'dnn_3',
    'dnn_4'
]

def run_model(model_type, checkpoint_path):
    
    model = models.create_model(model_type, checkpoint_path)
    model = model.to(device)
    
    # Train the Models
    if train:
        
        optimizer = optim.Adam
        loss_fn = nn.CrossEntropyLoss()
        
        models.train_model(model, trainloader_twisted, testloader, epochs, optimizer, loss_fn, device)
        
        models.save_model(model)
    
    ############ TESTING AND VALIDATION #######################
    
    if test:
    
        loss_fn = nn.CrossEntropyLoss()
        
        total_testing_examples, testing_accuracy, testing_loss = models.test_accuracy(model, testloader, loss_fn, device)

        print('//////////////////////////////// TESTING /////////////////////////////////////////////////////////////////////////////////////')
        print(f'{model.name} achieved {testing_accuracy} accuracy on {total_testing_examples} after {model.training_epochs} training epochs.')
        print('//////////////////////////////// TESTING /////////////////////////////////////////////////////////////////////////////////////\n\n')

def main():
    if is_all:
        for model in model_list:
            run_model(model, checkpoint_path)
    else:
        run_model(model_type, checkpoint_path)
        
    generate_figures.generate_figures()
    
if __name__=='__main__':
    main()
