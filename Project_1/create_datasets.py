import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 1_1 function definition
def func(x):
    return (np.exp(np.sin(x)))*np.cos(x)

def create_1_1_dataset():
    a = [['x', 'y']]
    for it in range(4000):
        x = round(random.uniform(-10.00, 10.00), 3)
        y = func(x)
        a.append([x, y])
    with open("./dataset_1_1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(a)
        
# if __name__=='__main__':
#     create_1_1_dataset()
#     test = pd.read_csv("./dataset_1_1.csv")
#     test = np.array(test)
#     # sort the array in ascending order by the first column to plot
#     test = test[test[:, 0].argsort()]
#     x = np.linspace(-10, 10, 1000)
#     plt.plot(x, func(x), color = 'blue')
#     plt.title("Line graph")
#     plt.xlabel("X axis")
#     plt.ylabel("Y axis")
#     plt.plot(test[:,0], test[:,1], color ="red")
#     plt.show()
#     print(test)
    