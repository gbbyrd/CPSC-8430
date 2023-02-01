import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import models
from torchsummary import summary

'''
This file will allow you to test the performance of multiple neural networks when trying
to approximate various non-linear functions
'''

chosen_function = 'cosine'

def plot_results(x_train, y_train, x_eval, model_predictions, epochs_to_converge):
    fig, axis = plt.subplots(1, 2)
    axis[0].plot(x_train, y_train, c='blue', linewidth=2.0, label='Ground Truth')
    axis[0].set_title(f'Model Predictions')
    axis[1].set_title(f'Model Loss vs Epochs')
    axis[0].set_xlabel(f'x')
    axis[0].set_ylabel(f'f(x)')
    axis[1].set_xlabel(f'epochs')
    axis[1].set_ylabel(f'loss')
    colors = ['red', 'green', 'blue']
    for count, pred in enumerate(model_predictions):
        epochs = epochs_to_converge[count][1:,0]
        loss = epochs_to_converge[count][1:,1]
        model_name = f'Model {count + 1}'
        axis[0].plot(x_eval, pred, c=colors[count], linewidth=0.5, label=model_name)
        axis[1].plot(epochs, loss, c=colors[count], label = model_name + " Loss")
        leg_pred = axis[0].legend()
        leg_loss = axis[1].legend()
    
    plt.show()
    
def test_functions(x):
        funcs = {
            'cosine': np.cos(x), # cosine function
            'sine': np.sin(x), # sine function
            'exponential': np.exp(x), # exponential function
            'power': np.power(x, 3), # cubic function
            'quadratic': np.power(x, 2) # quadratic
        }
        return funcs.get(chosen_function)
    
if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("We are using the following device for learning:", device)

    # Generate training data
    x_train = np.linspace(-5, 5, num=200).reshape(-1, 1)
    x_eval = np.linspace(-10, 10, num=200).reshape(-1, 1)
    y_train  = test_functions(x_train)

    # Input data
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    x_eval_tensor = torch.from_numpy(x_eval).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)


    # Define model
    model1 = models.network1(device=device)
    model2 = models.network2(device=device)
    model3 = models.network3(device=device)

    # Train models
    epochs = 10000

    result_eval1, epochs_to_converge_1 = models.train_model(model1, x_train_tensor, y_train_tensor, x_eval_tensor, epochs, 'model1.csv')
    result_eval2, epochs_to_converge_2 = models.train_model(model2, x_train_tensor, y_train_tensor, x_eval_tensor, epochs, 'model2.csv')
    result_eval3, epochs_to_converge_3 = models.train_model(model3, x_train_tensor, y_train_tensor, x_eval_tensor, epochs, 'model3.csv')
    
    model_predictions = [result_eval1]
    model_predictions.append(result_eval2)
    model_predictions.append(result_eval3)
    
    epochs_to_converge = [epochs_to_converge_1]
    epochs_to_converge.append(epochs_to_converge_2)
    epochs_to_converge.append(epochs_to_converge_3)
    
    summary(model1, (1, 1, 1))
    summary(model2, (1, 1, 1))
    summary(model3, (1, 1, 1))
    
    plot_results(x_train, y_train, x_eval, model_predictions, epochs_to_converge)
    
    
    
    


