import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import models

# Set GPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Set transform variable to transform data to normalized tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 10

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
    
    # Define epochs, optimizer, and loss function
    epochs = 5
    optimizer = optim.Adam
    loss_fn = nn.CrossEntropyLoss()
    
    # Train the Models
    models.train_model(model=model_cnn, dataloader=trainloader, epochs=epochs, 
                       optimizer=optimizer, loss_fn=loss_fn, device=device)
    models.train_model(model=model_dnn, dataloader=trainloader, epochs=epochs, 
                       optimizer=optimizer, loss_fn=loss_fn, device=device)
    
    
    
    

