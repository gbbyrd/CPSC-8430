import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import argparse

import models
import generate_figures

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False, help='Train the models and save checkpoints')

args = parser.parse_args()

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# define transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create the datasets and dataloaders

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

trainloader_model1 = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
trainloader_model2 = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                                 shuffle=True, num_workers=2)
trainloader_model3 = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                 shuffle=True, num_workers=2)
trainloader_model4 = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                 shuffle=True, num_workers=2)
trainloader_model5 = torch.utils.data.DataLoader(trainset, batch_size=512,
                                                 shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=2)

optimizer = optim.Adam
criterion = nn.CrossEntropyLoss()

# Train the same model for 10 epochs with different batch sizes
if args.train:
    epochs = 100
    model1 = models.DNN_10('interpolation_model1')
    model2 = models.DNN_10('interpolation_model2')
    model3 = models.DNN_10('interpolation_model3')
    model4 = models.DNN_10('interpolation_model4')
    model5 = models.DNN_10('interpolation_model5')
    
    model1 = model1.to(DEVICE)
    model2 = model2.to(DEVICE)
    model3 = model3.to(DEVICE)
    model4 = model4.to(DEVICE)
    model5 = model5.to(DEVICE)

    models.train_model(model1, trainloader_model1, testloader, epochs, optimizer,
                    criterion, DEVICE)
    models.train_model(model2, trainloader_model2, testloader, epochs, optimizer,
                    criterion, DEVICE)
    models.train_model(model3, trainloader_model3, testloader, epochs, optimizer,
                    criterion, DEVICE)
    models.train_model(model4, trainloader_model4, testloader, epochs, optimizer,
                    criterion, DEVICE)
    models.train_model(model5, trainloader_model5, testloader, epochs, optimizer,
                    criterion, DEVICE)

if not args.train:
    # Load the checkpoints
    model1 = torch.load('checkpoints/interpolation_model1.pth')
    model2 = torch.load('checkpoints/interpolation_model2.pth')
    model3 = torch.load('checkpoints/interpolation_model3.pth')
    model4 = torch.load('checkpoints/interpolation_model4.pth')
    model5 = torch.load('checkpoints/interpolation_model5.pth')

''' Flatness vs Generalization Part 1 '''

''' Get the parameters of the models, create new models with linear
interpolations of the parameters of the two models at ratios of -1 to 2
with a .1 step size '''

# Get interpolated model
models.analyze_interpolations(model1, model2, trainloader_model1, testloader,
                               criterion, DEVICE)

generate_figures.alpha_vs_accuracy()

''' Flatness vs Generalization Part 2 '''

model_list = [
    model1,
    model3,
    model4,
    model5,
    model2
]

# Careful!! Make sure this list corresponds to the batch sizes
# used in the dataloaders
batch_size_list = [
    32,
    64,
    256,
    512,
    1024
]

models.analyze_sensitivity(model_list, batch_size_list, criterion, 
                           trainloader_model1, testloader, DEVICE)

generate_figures.sensitivity_analysis()






