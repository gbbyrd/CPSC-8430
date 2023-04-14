import torch
import dcgan_model
import cv2
import numpy as np
import torchvision.utils as vutils

generator = dcgan_model.Generator()
generator.load_state_dict(torch.load('weights/netG_epoch_44.pth'))

noise_dim = 100
num_images = 100

for i in range(num_images):
    # generate random noise
    noise = torch.rand(1, noise_dim, 1, 1)
    
    fake = generator(noise)
    
    vutils.save_image(fake.detach(),'dcgan_images/fake_%03d.png' % (i),normalize=True)
    