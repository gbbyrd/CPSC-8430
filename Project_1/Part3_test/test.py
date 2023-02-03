import numpy as np

zeros = np.zeros(50)
new_zeros = np.random.choice(10, 50)
targets = [1, 2, 3, 4, 5, 6, 7]
labels = np.array(targets)
np.random.seed(12345)
mask = np.random.rand(len(labels)) <= 1.0
yes = 0


labels = np.array(targets)
np.random.seed(12345)
mask = np.random.rand(len(labels)) <= 1.0
rnd_labels = np.random.choice(10, mask.sum())
labels[mask] = rnd_labels
# we need to explicitly cast the labels from npy.int64 to
# builtin int type, otherwise pytorch will fail...
labels = [int(x) for x in labels]
targets = labels

yes = 0