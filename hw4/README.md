# Homework 4 - Deep Learning

## Description
The purpose of this work is to train two GAN models. The first GAN model is DCGAN while the second GAN model improves upon the first model by implementing techniques from WGANs and ACGANs.

## Run the code

### Model 1

To train the DCGAN, type the following command

`python3 dcgan_train.py`

This will output the saved version of the trained DCGAN in the dcgan_checkpoints directory and will output collections of images from various stages in the training process in the dcgan_output directory.

### Model 2

To train the WGAN/ACGAN, type the following command

`python3 wgan_acgan_train.py`

This will output the saved version of the WGAN/ACGAN model in the wgan_acgan_checkpoints directory and will output collections of images from various stages in the training process in the wgan_acgan_output directory.