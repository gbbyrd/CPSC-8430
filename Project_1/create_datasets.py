import csv
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Part1 import path_to_current_folder, data_folder

# 1_1 function definition
def func(x):
    return (np.exp(np.sin(x)))*np.cos(x)

def create_part1_dataset(dataset_name: str, num_data_points: int):
    # Create data folder if necessary
    if os.path.exists(os.path.join(path_to_current_folder, data_folder)):
        pass
    else:
        os.mkdir(os.path.join(path_to_current_folder, data_folder))
    
    a = [['x', 'y']]
    for it in range(num_data_points):
        x = round(random.uniform(-10.00, 10.00), 2)
        y = round(func(x), 2)
        a.append([x, y])
    csv_name = os.path.join(path_to_current_folder, dataset_name)
    with open(csv_name, "w", newline="") as f:
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
    