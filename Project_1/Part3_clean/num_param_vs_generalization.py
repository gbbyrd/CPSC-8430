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
import torchvision
import torchvision.transforms as transforms
import models

checkpoint_folder_path = 'checkpoints/'
    
# Set GPU device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

batch_size = 32

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                            transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size,
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

def run_model(model_name, checkpoint_path):
    model = models.create_model(model_name, checkpoint_path)

def main():
    pass

