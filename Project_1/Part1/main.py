import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import models

'''
This file will allow you to test the performance of 3 neural networks when trying
to approximate various non-linear functions
'''

chosen_function = 'quadratic'

def plot_results(x_train, y_train, model_predictions):
    for pred in model_predictions:
        plt.figure()
        plt.plot(x_train, y_train, c='blue', linewidth=1.0)
        plt.plot(x_eval, pred, c='red', linewidth=0.5)
        
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

    epochs = 10000

    result_eval1 = models.train_model(model1, x_train_tensor, y_train_tensor, x_eval_tensor, epochs)
    result_eval2 = models.train_model(model2, x_train_tensor, y_train_tensor, x_eval_tensor, epochs)
    result_eval3 = models.train_model(model3, x_train_tensor, y_train_tensor, x_eval_tensor, epochs)
    
    model_predictions = [result_eval1]
    model_predictions.append(result_eval2)
    model_predictions.append(result_eval3)
    
    plot_results(x_train, y_train, model_predictions)
    
    
    
    


