import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import os

def alpha_vs_accuracy():
    data = pd.read_csv('model_data/interpolated_model.csv').to_numpy()
    fig,ax = plt.subplots()
    ax.plot(data[:,0], data[:,2], 'b', linestyle='dotted', label='test') # plot testing loss
    ax.plot(data[:,0], data[:,4], 'b', label='train') # plot training loss
    ax.set_xlabel('alpha')
    ax.set_ylabel('loss', color='blue')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(data[:,0], data[:,1], 'r', linestyle='dotted') # plot testing accuracy
    ax2.plot(data[:,0], data[:,3], 'r') # plot training accuracy
    ax2.set_ylabel('accuracy', color='red')
    
    fig.savefig('figures/flatness_vs_generalization1.png')
    
def sensitivity_analysis():
    data = pd.read_csv('model_data/sensitivity_analysis.csv').to_numpy()
    print(data)
    fig,ax = plt.subplots(1, 2, figsize=(16,6))
    plt.xscale('log')
    ax_0 = ax[0].twinx()
    ax_1 = ax[1].twinx()
    ax[0].plot(data[:,0], data[:,3], 'b', linestyle='dotted', label='test') # add testing loss
    ax[0].plot(data[:,0], data[:,5], 'b', label='train') # add training loss
    ax[0].set_xlabel('batch_size')
    ax[0].set_ylabel('loss', color='blue')
    ax_0.plot(data[:,0], data[:,1], 'r', label='sensitivity') # add sensitivity
    ax_0.set_ylabel('sensitivity', color='red')
    ax[0].legend()
    ax_0.legend()
    ax[1].plot(data[:,0], data[:,2], 'b', linestyle='dotted', label='test') # add testing accuracy
    ax[1].plot(data[:,0], data[:,4], 'b', label='train') # add training accuracy
    ax[1].set_xlabel('batch_size')
    ax[1].set_ylabel('accuracy', color='blue')
    ax_1.plot(data[:,0], data[:,1], 'r', label='sensitivity') # add sensitivity
    ax_1.set_ylabel('sensitivity', color='red')
    ax[1].legend()
    ax_1.legend()
    
    fig.savefig('figures/sensitivity_analysis.png')

    

def random_fit():
    ''' Generates figure for the random fit experiment'''
    if not os.path.exists('figures/'):
        os.mkdir('figures/')
        
    csv_files = glob.glob('model_data/*.csv')
    csv_name = ''
    for name in csv_files:
        if 'random' in name:
            csv_name = name
    fig, axis = plt.subplots(1, 2)
    data = pd.read_csv(csv_name)
    data = np.array(data)
    epochs = data[:, 0]
    training_loss = data[:, 1]
    training_accuracy = data[:, 2]
    testing_loss = data[:, 3]
    testing_accuracy = data[:, 4]
    axis[0].plot(epochs, training_loss, c='blue', label='train loss')
    axis[0].plot(epochs, testing_loss, c='red', label='test loss')
    axis[1].plot(epochs, training_accuracy, c='blue', label='train accuracy')
    axis[1].plot(epochs, testing_accuracy, c='red', label='test accuracy')
    axis[0].set_xlabel('epochs')
    axis[0].set_ylabel('loss')
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('accuracy')
    axis[0].set_title(csv_name[:-4] + ' loss')
    axis[1].set_title(csv_name[:-4] + ' accuracy')
    axis[0].legend()
    axis[1].legend()
    plt.savefig(os.path.join('figures/', csv_name[11:-4]))
    
            

def generate_figures():
    # # group CNN and DNN models together
    csv_files = glob.glob('model_data/*csv')
    cnn_files = []
    dnn_files = []
    for name in csv_files:
        if 'cnn' in name:
            cnn_files.append(name)
        else:
            dnn_files.append(name)
    for csv_name in cnn_files:
        fig, axis = plt.subplots(1, 2)
        data = pd.read_csv(csv_name)
        data = np.array(data)
        epochs = data[:, 0]
        training_loss = data[:, 1]
        training_accuracy = data[:, 2]
        testing_loss = data[:, 3]
        testing_accuracy = data[:, 4]
        axis[0].plot(epochs, training_loss, c='blue', label='train loss')
        axis[0].plot(epochs, testing_loss, c='red', label='test loss')
        axis[1].plot(epochs, training_accuracy, c='blue', label='train accuracy')
        axis[1].plot(epochs, testing_accuracy, c='red', label='test accuracy')
        axis[0].set_xlabel('epochs')
        axis[0].set_ylabel('loss')
        axis[1].set_xlabel('epochs')
        axis[1].set_ylabel('accuracy')
        axis[0].set_title(csv_name[11:-4] + ' loss')
        axis[1].set_title(csv_name[11:-4] + ' accuracy')
        axis[0].legend()
        axis[1].legend()
        plt.savefig(os.path.join('figures/', csv_name[11:-3]))
    fig, axis = plt.subplots(1, 2)
    color_array = ['blue', 'red', 'green', 'yellow', 'black', 'orange', 'pink', 'brown', 'purple', 'aqua']
    for model, csv_name in enumerate(cnn_files):
        data = pd.read_csv(csv_name)
        data = np.array(data)
        epochs = data[:, 0]
        training_loss = data[:, 1]
        training_accuracy = data[:, 2]
        testing_loss = data[:, 3]
        testing_accuracy = data[:, 4]
        axis[0].plot(epochs, training_loss, c=color_array[model], label=f'cnn_{model} train loss')
        axis[0].plot(epochs, testing_loss, c=color_array[model+3], label=f'cnn_{model} test loss')
        axis[1].plot(epochs, training_accuracy, c=color_array[model], label=f'cnn_{model} train acc')
        axis[1].plot(epochs, testing_accuracy, c=color_array[model+3], label=f'cnn_{model} test acc')
    axis[0].set_xlabel('epochs')
    axis[0].set_ylabel('loss')
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('accuracy')
    axis[0].set_title('CNN model loss comparison')
    axis[1].set_title('CNN model accuracy comparison')
    axis[0].legend()
    axis[1].legend()
    plt.savefig('figures/CNN_compare.png')

    # DNN Models
    for csv_name in dnn_files:
        fig, axis = plt.subplots(1, 2)
        data = pd.read_csv(csv_name)
        data = np.array(data)
        epochs = data[:, 0]
        training_loss = data[:, 1]
        training_accuracy = data[:, 2]
        testing_loss = data[:, 3]
        testing_accuracy = data[:, 4]
        axis[0].plot(epochs, training_loss, c='blue', label='train loss')
        axis[0].plot(epochs, testing_loss, c='red', label='test loss')
        axis[1].plot(epochs, training_accuracy, c='blue', label='train acc')
        axis[1].plot(epochs, testing_accuracy, c='red', label='test acc')
        axis[0].set_title(csv_name[11:-4] + ' loss')
        axis[1].set_title(csv_name[11:-4] + 'accuracy')
        axis[0].set_xlabel('epochs')
        axis[0].set_ylabel('loss')
        axis[1].set_xlabel('epochs')
        axis[1].set_ylabel('accuracy')
        axis[0].legend()
        axis[1].legend()
        plt.savefig(os.path.join('figures/', csv_name[11:-3]))
    fig, axis = plt.subplots(1, 2)
    for model, csv_name in enumerate(dnn_files): 
        data = pd.read_csv(csv_name)
        data = np.array(data)
        epochs = data[:, 0]
        training_loss = data[:, 1]
        training_accuracy = data[:, 2]
        testing_loss = data[:, 3]
        testing_accuracy = data[:, 4]
        axis[0].plot(epochs, training_loss, c=color_array[model], label=f'dnn_{model} train loss')
        axis[0].plot(epochs, testing_loss, c=color_array[model+5], label=f'dnn_{model} test loss')
        axis[1].plot(epochs, training_accuracy, c=color_array[model], label=f'dnn_{model} train acc')
        axis[1].plot(epochs, testing_accuracy, c=color_array[model+5], label=f'dnn_{model} test acc')
    axis[0].legend()
    axis[1].legend()
    axis[0].set_xlabel('epochs')
    axis[0].set_ylabel('loss')
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('accuracy')
    axis[0].set_title('DNN model loss comparison')
    axis[1].set_title('DNN model accuracy comparison')
    plt.savefig(os.path.join('figures/DNN_compare.png'))
