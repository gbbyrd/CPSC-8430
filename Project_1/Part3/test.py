import numpy as np

arr = np.arange(9)
np.random.shuffle(arr)

arr = np.arange(100).reshape(10, 10)
print(arr)

np.random.shuffle(arr[:, 2])
print(arr)