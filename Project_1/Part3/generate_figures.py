import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import os

def generate_figures():
    # group CNN and DNN models together
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
    color_array = ['blue', 'red', 'green', 'yellow', 'black', 'orange']
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
        axis[0].plot(epochs, testing_loss, c=color_array[model+3], label=f'dnn_{model} test loss')
        axis[1].plot(epochs, training_accuracy, c=color_array[model], label=f'dnn_{model} train acc')
        axis[1].plot(epochs, testing_accuracy, c=color_array[model+3], label=f'dnn_{model} test acc')
    axis[0].legend()
    axis[1].legend()
    axis[0].set_xlabel('epochs')
    axis[0].set_ylabel('loss')
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('accuracy')
    axis[0].set_title('DNN model loss comparison')
    axis[1].set_title('DNN model accuracy comparison')
    plt.savefig(os.path.join('figures/DNN_compare.png'))