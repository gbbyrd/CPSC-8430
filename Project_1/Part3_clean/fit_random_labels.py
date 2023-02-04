'''
Grayson Byrd
CPSC 8430 - Deep Learning
Part 3 - Can a model fit random labels?

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

checkpoint_folder_path = 'checkpoints/'
    
# Set GPU device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Hardcode hyperparameters
batch_size = 32
epochs = 300

class MNIST10RandomLabels(torchvision.datasets.MNIST):
  """MNIST10 dataset, with support for randomly corrupt labels."""
  def __init__(self, num_classes=10, **kwargs):
    super(MNIST10RandomLabels, self).__init__(**kwargs)
    labels = np.random.choice(10, len(self.targets))
    labels = [int(x) for x in labels]
    self.targets = labels
    
# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset_twisted = MNIST10RandomLabels(root='./data', train=True, download=True,
                            transform=transform)

# print(trainset_twisted[0][0])
    
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

trainloader_twisted = torch.utils.data.DataLoader(trainset_twisted, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def main():
    # Create model
    random_fit_model = models.create_model('dnn_random_fit')
    random_fit_model.to(DEVICE)
    
    # Train the model
    optimizer = optim.Adam
    loss_fn = nn.CrossEntropyLoss()
    
    models.train_model(random_fit_model, trainloader_twisted, testloader, 
                        epochs, optimizer, loss_fn, DEVICE)
    
    models.save_model(random_fit_model)
    
    generate_figures.random_fit()
    
if __name__ == '__main__':
    main()

