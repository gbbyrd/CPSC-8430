import torch
import torch.nn as nn
import numpy as np
from alt_seq2seq_model import EncoderRNN, DecoderRNN
from dataset import HW2_Dataset

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
    for idx in range(min(reference_length, candidate_length)):
        # Ensure that the tensors are both on the cpu
        predicted[idx] = predicted[idx].to('cpu')
        target[idx] = target[idx].to('cpu')
        if predicted[idx] == target[idx]:
            correct += 1
    
    precision = correct / candidate_length
    
    bleu = BP * precision
    
    return bleu

def decode_sentence(sentence, num_to_word_dict):
    decoded_sentence = []
    for num in sentence:
        num = num.detach().item()
        word = num_to_word_dict[num]
        decoded_sentence.append(word)
        
    return decoded_sentence
        
def test(input, target, encoder, decoder, criterion, device, max_length=25):

    input = input.to(device)
    target = target.to(device)
    
    input_length = input.size(0)
    target_length = target.size(0)
    
    encoder_top_hidden, encoder_bottom_hidden = encoder.init_hidden()
    
    encoder_outputs = torch.zeros(input_length, encoder.dim_hidden, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_top_hidden, encoder_bottom_hidden = encoder(input[0], encoder_top_hidden, 
                                                 encoder_bottom_hidden)
    
    decoder_top_hidden, decoder_bottom_hidden = encoder_top_hidden, encoder_bottom_hidden
    
    decoder_prev_bottom_output = torch.tensor([0], device=device)
    
    sentence = []
    
    for di in range(target_length):
        decoder_top_hidden, decoder_bottom_hidden, decoder_bottom_output = decoder(decoder_prev_bottom_output, decoder_top_hidden,
                                                                                decoder_bottom_hidden)
        loss += criterion(decoder_bottom_output, target[di])
        topv, topi = torch.topk(decoder_bottom_output, 1)
        decoder_prev_bottom_output = topi.squeeze(0)
        sentence.append(topi.squeeze().detach())
        if topi.squeeze().detach() == 1:
            break
    
    return loss.item() / target_length, sentence

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    
    testset = HW2_Dataset(train=True)
    MAX_LENGTH = 25
    
    encoder_checkpoint = 'checkpoints/encoder_130.pth'
    decoder_checkpoint = 'checkpoints/decoder_130.pth'
    
    encoder_state_dict = torch.load(encoder_checkpoint)
    decoder_state_dict = torch.load(decoder_checkpoint)
    
    encoder = EncoderRNN(testset.dict_size, MAX_LENGTH, dim_hidden=256, dim_word=256)
    decoder = DecoderRNN(testset.dict_size, MAX_LENGTH, dim_hidden=256, dim_word=256)
    
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    
    criterion = nn.NLLLoss()
    
    # Loop through all testing videos and get predicted sentences
    pred_sentences = []
    target_sentences = []
    for idx in range(len(testset)):
        feats, caption = testset[idx]
        loss, pred = test(feats, caption, encoder, decoder, criterion, DEVICE, MAX_LENGTH)
        pred_sentences.append(pred)
        target_sentences.append(caption)
        
    bleu_scores = []
    for idx in range(len(testset)):
        bleu_score = evaluate(pred_sentences[idx], target_sentences[idx])
        bleu_scores.append(bleu_score)
    
    final_bleu_evaluation_score = sum(bleu_scores) / len(bleu_scores)
    
    print(f'BLEU Performance Evaluation: {final_bleu_evaluation_score}')
    
    