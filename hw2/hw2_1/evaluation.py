import torch
import numpy as np

def evaluate(predicted, target):
    """This is an implementation of the BLEU@1 evaluation for video captioning models.

    Args:
        predicted (Tensor): A tensor holding the predicted words of the caption.
        target (Tensor): A tensor of the ground truth caption.
    """
    
    candidate_length = len(predicted)
    reference_length = len(target)
    
    BP = ...
    
    if candidate_length > reference_length:
        BP = 1
    else:
        BP = np.exp((1-reference_length)/candidate_length)
        
    correct = 0
    for idx in range(reference_length):
        if predicted[idx] == target[idx]:
            correct += 1
    
    precision = correct / candidate_length
    
    bleu = BP * precision
    
    return bleu