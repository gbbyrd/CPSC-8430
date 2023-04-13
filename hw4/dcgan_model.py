import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_noise_dim=100, embedding_dim=256):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(10, embedding_dim)
        self.architecture = nn.Sequential(
            # nn.Linear(in_noise_dim+embedding_dim, 4*4*512),
            nn.Linear(in_noise_dim, 4*4*512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 4, 4)),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, text=None):
        # text_embedding = self.embedding(text)
        # input = torch.cat((noise, text_embedding), dim=1)
        return self.architecture(noise)
    
class Discriminator(nn.Module):
    def __init__(self, image_dim=64*64*3, embedding_dim=256):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10, embedding_dim)
        self.architecture = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256, momentum=.9),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512, momentum=.9),
            # supposed to do leaky relu and then concatenate? ask the professor
            nn.Conv2d(512, 512, kernel_size=1, stride=(1,1)),
            nn.Flatten(),
            nn.Linear(4608, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        return self.architecture(image)
    
