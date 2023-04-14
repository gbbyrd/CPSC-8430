import torch
import torchvision
from torchvision import transforms, datasets
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import dcgan_model

args = ArgumentParser()

args.add_argument('--train', action='store_true', default=False, help='Specify training')
args.add_argument('--batch_size', action='store', type=int, default=32, help='Specify the batch size for training')
args.add_argument('--epochs', action='store', type=int, default=100, help='Specify the number of epochs you want to train for')
args.add_argument('--save_every', action='store', type=int, default=5, help='Specify number of epochs before saving')

def train(device, generator, discriminator, criterion, optimizer_g, optimizer_d, epochs, trainloader, save_every):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    fixed_noise = torch.randn(128, 100, 1, 1, device=device)
    
    for epoch in range(epochs):
        
        d_avg_loss = 0
        g_avg_loss = 0
        
        for i, (images_real, label) in enumerate(trainloader):
            images_real = images_real.to(device)
 
            discriminator.zero_grad()
            
            real = images_real.to(device)
            
            batch_size = len(images_real)
            
            # create labels
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)
            
            # get discriminator predictions
            d_pred_real = discriminator(images_real)
            D_x = d_pred_real.mean().item()
            
            # calculate the loss from real images
            d_loss_real = criterion(d_pred_real, labels_real)  
            d_loss_real.backward()
            
            # generate fake images for training
            noise = torch.rand(batch_size, 100, 1, 1).to(device)
            
            images_fake = generator(noise)
            
            # get discriminator predictions
            '''this .detach() here is crucial as it prevents the gradients for 
            the generator from being calculated. This would cause an error later on
            in the code when we try to .backward() through the loss calculated for 
            the generator because the gradients were calculated twice on the same
            parameters before an optimizer step was called'''
            d_pred_fake = discriminator(images_fake.detach())
            D_G_z1 = d_pred_fake.mean().item()
            
            # calculate the loss from fake images
            d_loss_fake = criterion(d_pred_fake, labels_fake)
            d_loss_fake.backward()
            
            # compute gradients
            d_loss = d_loss_real + d_loss_fake
            
            d_avg_loss += round(d_loss.item() / (batch_size * 2), 3)
            
            # update weights
            optimizer_d.step()
            
            ####################################################################
            # TRAIN THE GENERATOR
            ####################################################################
            
            generator.zero_grad()
            
            d_pred_fake = discriminator(images_fake)
            D_G_z2 = d_pred_fake.mean().item()
            
            g_loss = criterion(d_pred_fake, labels_real)
            
            g_loss.backward()
            
            optimizer_g.step()
            
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(trainloader), d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            
            g_avg_loss += round(g_loss.item() / (batch_size), 3)
            
            #save the output
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real,'output/real_samples.png',normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
            
        d_loss = round(d_avg_loss / (len(trainloader) * 2), 3)
        g_loss = round(g_avg_loss / len(trainloader), 3)
        
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Epoch: {epoch}")
        print("{:<20} {:<10}".format('Discriminator Loss:', d_loss))
        print("{:<20} {:<10}".format('Generator Loss:', g_loss))
        print("\n")
        
        if epoch % save_every == 0:
            torch.save(discriminator.state_dict(), f'checkpoints/D_{epoch}.pth')
            torch.save(generator.state_dict(), f'checkpoints/G_{epoch}.pth')
            
def main():
    
    arguments = args.parse_args()
    
    # Set GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Working on {device}...')
    
    batch_size = arguments.batch_size

    # Set transform variable to transform data to normalized tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    cifar10_dictionary = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    
    generator = dcgan_model.Generator()
    
    discriminator = dcgan_model.Discriminator()
    
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train(device, generator, discriminator, criterion, optimizer_g, optimizer_d, arguments.epochs, trainloader, arguments.save_every)

if __name__ == '__main__':
    main()