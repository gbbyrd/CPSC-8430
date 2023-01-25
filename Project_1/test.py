from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time

data_test = pd.read_csv('data/test_dataset_1_1.csv')
list = [[1,1],
        [3,4],
        [5,4]]
list = np.array(list)

data_test = data_test.to_numpy()
# print(data_test[:, 0], data_test[:, 1])

for i in range(100):
    print(i)
    time.sleep(2)