import torch
import torch.nn as nn
from dataset import HW2_Dataset
from alt_seq2seq_model import EncoderRNN, DecoderRNN

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
    
    MAX_LENGTH = 25

    trainset = HW2_Dataset(train=True)

    encoder = EncoderRNN(trainset.dict_size, MAX_LENGTH, dim_hidden=256, dim_word=256)
    decoder = DecoderRNN(trainset.dict_size, MAX_LENGTH, dim_hidden=256, dim_word=256)
    encoder_state_dict = torch.load('checkpoints/encoder_200.pth')
    decoder_state_dict = torch.load('checkpoints/decoder_200.pth')

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    encoder.eval()
    decoder.eval()
    
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    
    criterion = nn.NLLLoss()

    pred_sentences = []
    target_sentences = []
    losses = []
    
    for i in range(10):
        feats, caption = trainset[i]
        
        decoded_caption = decode_sentence(caption, trainset.idx_to_word)
        target_sentences.append(decoded_caption)
        
        loss, pred_sentence = test(feats, caption, encoder, decoder, criterion, DEVICE, MAX_LENGTH)
        
        losses.append(loss)
        
        pred_sentence = decode_sentence(pred_sentence, trainset.idx_to_word)
        
        pred_sentences.append(pred_sentence)
    
    for idx in range(len(pred_sentences)):
        print(f'Target: {target_sentences[idx]}')
        print(f'Predicted: {pred_sentences[idx]}')
        print(f'Loss: {losses[idx]}')
        
    
    