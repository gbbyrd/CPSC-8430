import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--batch_size', action='store', default=32, type=int, help='Batch size for testing and training the model')
parser.add_argument('--epochs', action='store', default=5, type=int, help='Number of epochs to train the model for')
parser.add_argument('--checkpoint', action='store', default=None, type=str, help='.pth file to load into the model')
parser.add_argument('--model_type', action='store', default=None, type=str, help='Specify the model class')
parser.add_argument('--all_models', action='store_true', default=False, help='Loop through all models')
