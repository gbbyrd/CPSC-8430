import torch
import torch.nn as nn

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

nc=3

class Generator_1(nn.Module):
    def __init__(self, ngpu=1):
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
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output

class Discriminator_1(nn.Module):
    def __init__(self, ngpu=1):
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
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

# generator architecture from paper @ https://arxiv.org/pdf/1511.06434.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convT1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(1024, momentum=0.9)
        self.relu = nn.ReLU(True)
        
        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(256, momentum=0.9)
        
        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(128, momentum=0.9)
        
        self.convT4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(64, momentum=0.9)
        
        self.convT5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, noise):
        x = self.convT1(noise)
        x = self.bnorm1(x)
        x = self.relu(x)
    
        x = self.convT2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        
        x = self.convT3(x)
        x = self.bnorm3(x)
        x = self.relu(x)
        
        x = self.convT4(x)
        x = self.bnorm4(x)
        x = self.relu(x)
        
        x = self.convT5(x)
        x = self.tanh(x)
        # final size [batch_size, 3, 64, 64]
        
        return x

# discriminator will just reverse the architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(64, momentum=0.9)
        self.leakyRelu = nn.LeakyReLU(0.02, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(128, momentum=0.9)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(512, momentum=0.9)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(512, momentum=0.9)
        
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image):
        x = self.conv1(image)
        x = self.bnorm1(x)
        x = self.leakyRelu(x)
        
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.leakyRelu(x)
        
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.leakyRelu(x)
        
        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.leakyRelu(x)
        
        x = self.conv5(x)
        x = self.sigmoid(x)
        
        return x.view(-1, 1)
