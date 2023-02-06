import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', default=True, help='Train the models')
parser.add_argument('--test', action='store_true', default=True, help='Test the model on the test set at the end of training')
parser.add_argument('--epochs', action='store', default=10, type=int, help='Number of epochs to run')
parser.add_argument('--batch_size', action='store', default=32, type=int, help='Batch size for the training algorithm')
parser.add_argument('--model_type', action='store', default=None, type=str, help='If training only one model, specify the type to train')
parser.add_argument('--checkpoint', action='store', type=str, default=None, help='Load the .pth file specified after this argument')
parser.add_argument('--all_models', action='store_true', default=False, help='Trains all of the models in a loop.')