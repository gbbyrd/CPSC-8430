import torch
import dcgan_model

noise = torch.rand(32, 100, 1, 1)

generator = dcgan_model.Generator()
discriminator = dcgan_model.Discriminator()

fake = generator(noise)

pred = discriminator(fake)

