import torch
import torch.nn as nn

class Generator_1(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator_1, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, input):
        output = self.main(input)
        return output

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
    
class Discriminator_1(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator_1, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
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
    
