import torch
import torch.nn as nn

# generator architecture from paper @ https://arxiv.org/pdf/1511.06434.pdf
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convT1 = nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(1024, momentum=0.9)
        self.relu = nn.ReLU(True)
        
        self.convT2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(512, momentum=0.9)
        
        self.convT3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(256, momentum=0.9)
        
        self.convT4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(128, momentum=0.9)
        
        self.convT5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(128, momentum=0.9)
        self.leakyRelu = nn.LeakyReLU(0.02, inplace=True)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(256, momentum=0.9)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(512, momentum=0.9)
        
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(1024, momentum=0.9)
        
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False)
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
