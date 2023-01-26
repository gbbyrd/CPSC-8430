'''
Grayson Byrd

Part 2 of Project 1

This file trains a convolutional and deep neural network on the CIFAR-10
dataset and compares their performance and training times. 
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

# Save the trained models
checkpoint_paths = os.listdir('Project_1/Part2/checkpoints')

exit()

arguments = args.parser.parse_args()

# Set GPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = arguments.batch_size

trainset = torchvision.datasets.CIFAR10(root='./Project_1/Part2/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./Project_1/Part2/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# function to depict and image
def show_image(img):
    img = img / 2 + 0.5
    numpy_img = img.numpy()
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()

if __name__=='__main__':
    ##### Uncomment for image troubleshooting #####
    # # get random training images
    # iterator = iter(trainloader)
    # rand_images, rand_labels = next(iterator)
    # print(rand_images.size())
    # # show images
    # show_image(torchvision.utils.make_grid(rand_images))
    # # print labels
    # print(' '.join(f'{classes[rand_labels[j]]:5s}' for j in range(batch_size)))
    
    # Inititiate Models
    model_cnn = models.CNN().to(device)
    model_dnn = models.DNN().to(device)
    
    # Train the Models
    if arguments.train:
        
        # If there is a saved model, load the saved model
        
        
        # Define epochs, optimizer, and loss function
        epochs = arguments.epochs
        optimizer = optim.Adam
        loss_fn = nn.CrossEntropyLoss()
        
        models.train_model(model=model_cnn, dataloader=trainloader, epochs=epochs, 
                        optimizer=optimizer, loss_fn=loss_fn, device=device)
        models.train_model(model=model_dnn, dataloader=trainloader, epochs=epochs, 
                        optimizer=optimizer, loss_fn=loss_fn, device=device)
        
        # Reflect the updated number of training epochs inside the model classes
        model_cnn.training_epochs += epochs
        model_dnn.training_epochs += epochs
    
        # Save the trained models
        checkpoint_paths = os.listdir('Project_1/Part2/checkpoints')
        
        PATH = 'Project_1/Part2/checkpoints'
        os.mkdir(PATH)
        torch.save(model_cnn.state_dict(), PATH + f'/{model_cnn.training_epochs + epochs}_model_cnn.pth')
        torch.save(model_dnn.state_dict(), PATH + f'/{model_dnn.training_epochs + epochs}_model_dnn.pth')
    
    ############ TESTING AND VALIDATION #######################
    
    # # Load the models
    # model_cnn = models.CNN()
    # model_cnn.load_state_dict(torch.load(PATH + '/model_cnn.pth'))
    
    # model_dnn = models.DNN()
    # model_dnn.load_state_dict(torch.load(PATH + '/model_dnn.pth'))
    
    # Test accuracy of the trained models
    
    models.test_accuracy(model_cnn, testloader, batch_size, device)
    models.test_accuracy(model_dnn, testloader, batch_size, device)
    
    
    
    

