import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--batch_size', action='store', default=10, type=int, help='Batch size for testing and training the model')
parser.add_argument('--epochs', action='store', default=5, type=int, help='Number of epochs to train the model for')