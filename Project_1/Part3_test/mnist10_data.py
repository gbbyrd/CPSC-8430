"""
mnist-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class MNIST10RandomLabels(datasets.MNIST):
  """MNIST10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, num_classes=10, **kwargs):
    super(MNIST10RandomLabels, self).__init__(**kwargs)
    labels = np.random.choice(10, len(self.targets))
    labels = [int(x) for x in labels]
    self.targets = labels
    
    