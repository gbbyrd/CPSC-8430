import torch
import torchvision
from torchvision import transforms, datasets
from argparse import ArgumentParser
import dcgan_model

args = ArgumentParser()

args.add_argument('--train', action='store_true', default=False, help='Specify training')
args.add_argument('--batch_size', action='store', default=32, help='Specify the batch size for training')
args.add_argument('--epochs', action='store', default=100, help='Specify the number of epochs you want to train for')
args.add_argument('--save_every', action='store', default=5, help='Specify number of epochs before saving')

def train(generator, discriminator, criterion, optimizer_g, optimizer_d, epochs, trainloader, save_every):
    for epoch in epochs:
        
        d_avg_loss = 0
        g_avg_loss = 0
        
        for images_real, label in trainloader:
            
            ####################################################################
            # TRAIN THE DISCRIMINATOR FIRST
            ####################################################################
            
            discriminator.zero_grad()
            
            batch_size = len(images_real)
            
            # create labels
            labels_real = torch.ones(batch_size, 1)
            labels_fake = torch.zeros(batch_size, 1)
            
            # get discriminator predictions
            d_pred_real = discriminator(images_real)
            
            # calculate the loss from real images
            d_loss_real = criterion(d_pred_real, labels_real)  
            
            # generate fake images for training
            noise = torch.rand(batch_size, 100)
            
            images_fake = generator(noise)
            
            # get discriminator predictions
            d_pred_fake = discriminator(images_fake)
            
            # calculate the loss from fake images
            d_loss_fake = criterion(d_pred_fake, labels_fake)
            
            # compute gradients
            d_total_loss = d_loss_real + d_loss_fake
            d_total_loss.backward()
            
            d_avg_loss += round(d_total_loss / (batch_size * 2), 3)
            
            # update weights
            optimizer_d.step()
            
            ####################################################################
            # TRAIN THE GENERATOR
            ####################################################################
            
            generator.zero_grad()
            
            d_pred_fake = discriminator(images_fake)
            
            g_loss = criterion(d_pred_fake, labels_real)
            
            g_loss.backward()
            
            optimizer_g.step()
            
            g_avg_loss += round(g_loss / (batch_size), 3)
            
        d_loss = round(d_avg_loss / (len(trainloader * 2)), 3)
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
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

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
    
if __name__ == '__main__':
    main()
