import torch
import torchvision
from torch import optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import args
import models

''' The purpose of this file is to depic the optimization process. This
file collects the weights of a DNN trained on the MNIST dataset every
3 epochs. The dimension of the weights is reduced to 2 by PCA and then
plotted on the graph to visualize the optimization process.'''

# Parse arguments
arguments = args.parser.parse_args()

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [transforms.ToTensor()]
)

# Load the datasets
trainset = torchvision.datasets.MNIST('./data', train=True, 
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', train=False,
                                     download=True, transform=transform)

# Load the dataloaders
trainloader = DataLoader(trainset, batch_size=arguments.batch_size,
                         shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=arguments.batch_size,
                        shuffle=False, num_workers=2)

def main():
    
    # Create model, define optimizers and criterion
    model = models.DNN_1()
    
    optimizer = optim.Adam
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    models.train_model_weight_pca(model, trainloader, testloader, arguments.epochs,
                                  optimizer, criterion, device)
    
if __name__ == '__main__':
    main()
    
    