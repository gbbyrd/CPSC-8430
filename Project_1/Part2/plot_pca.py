import pandas as pd
import matplotlib.pyplot as plt

'''This script reads the values from the model_data/pca_analysis.csv file and 
creates a scatter plot of the data.

The same model was trained a total of 8 times. Each time with 51 epochs, and the
first layer weights were collected every 3 epochs.
'''

# Get the pca data
pca_data = pd.read_csv('model_data/pca_analysis.csv', header=None)
pca_data = pca_data.to_numpy()
print(type(pca_data))
print(pca_data.shape)

color_array = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple', 'brown']
color_itr = iter(color_array)
color = ...
count = 17
for i in range(len(pca_data)):
    if count == 17:
        count = 0
        color = next(color_itr)
    count += 1
    plt.scatter(pca_data[i,0], pca_data[i,1], c=color)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Analysis of First Layer Weights')

plt.savefig('figures/pca_analysis.png')



