import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import HW2_Dataset
import random

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 25
trainset = HW2_Dataset(train=True)

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=4096, bos_id=0, 
                 eos_id=1, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.rnn1 = nn.LSTM(dim_vid, dim_hidden, n_layers, batch_first=True)
        self.rnn2 = nn.LSTM(dim_hidden + dim_word, dim_hidden, n_layers, batch_first=True)

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.n_layers = n_layers
        # self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        
    def forward(self, vid_feat, top_hidden, bottom_hidden):
        
        padding_word = torch.zeros([1, self.dim_word]).cuda()
        vid_feat = vid_feat.view(1, -1).type(torch.FloatTensor).cuda()
        
       
        # Pass through top LSTM
        top_output, top_hidden = self.rnn1(vid_feat, top_hidden)
        
        # Concatenate top output with the padded word
        bottom_input = torch.cat((top_output, padding_word), dim=1)
        
        # Pass through bottom LSTM
        bottom_output, bottom_hidden = self.rnn2(bottom_input, bottom_hidden)
        
        return top_hidden, bottom_hidden
    
    def init_hidden(self):
        top_hidden = None
        bottom_hidden = None
        return top_hidden, bottom_hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=4096, bos_id=0, 
                 eos_id=1, n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.rnn1 = nn.LSTM(dim_vid, dim_hidden, n_layers, batch_first=True)
        self.rnn2 = nn.LSTM(dim_hidden + dim_word, dim_hidden, n_layers, batch_first=True)

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, prev_bottom_output, top_hidden, bottom_hidden):
        
        padding_video = torch.zeros([1, self.dim_vid]).cuda()
        
        # Embed predicted word to hidden dim space then use nonlinear activation
        prev_bottom_output = self.embedding(prev_bottom_output)
        prev_bottom_output = F.relu(prev_bottom_output)
        
        # Pass through top LSTM
        top_output, top_hidden = self.rnn1(padding_video, top_hidden)
        # Concatenate top output with the padded word
        prev_bottom_output = prev_bottom_output
        bottom_input = torch.cat((top_output, prev_bottom_output), dim=1)
        
        # Pass through bottom LSTM
        bottom_output, bottom_hidden = self.rnn2(bottom_input, bottom_hidden)
        
        # Pass bottom_output through linear layer to get in dim of vocab length
        bottom_output = self.out(bottom_output)
        
        # Pass bottom_output through softmax layer
        bottom_output = self.softmax(bottom_output)
        
        return top_hidden, bottom_hidden, bottom_output
    
    def init_hidden(self):
        top_hidden = torch.zeros(1, 1, self.dim_hidden, device=device)
        bottom_hidden = torch.zeros(1, 1, self.dim_hidden, device=device)
        return top_hidden, bottom_hidden
    
    # def init_SOS_token(self):
    #     sos_token = torch.zeros(1, 1, self.dim_output)
    #     sos_token[0, 0, 0] = 1
    #     return sos_token
    
def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):

    input = input.to(device)
    target = target.to(device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input.size(0)
    target_length = target.size(0)
    
    encoder_top_hidden, encoder_bottom_hidden = encoder.init_hidden()
    
    encoder_outputs = torch.zeros(input_length, encoder.dim_hidden, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_top_hidden, encoder_bottom_hidden = encoder(input[0], encoder_top_hidden, 
                                                 encoder_bottom_hidden)
    
    decoder_top_hidden, decoder_bottom_hidden = encoder_top_hidden, encoder_bottom_hidden
    
    decoder_prev_bottom_output = torch.tensor([SOS_token], device=device)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_top_hidden, decoder_bottom_hidden, decoder_bottom_output = decoder(decoder_prev_bottom_output, decoder_top_hidden,
                                                                                        decoder_bottom_hidden)
            loss += criterion(decoder_bottom_output, target[di])
            decoder_prev_bottom_output = target[di]
        
    else:
        for di in range(target_length):
            decoder_top_hidden, decoder_bottom_hidden, decoder_bottom_output = decoder(decoder_prev_bottom_output, decoder_top_hidden,
                                                                                    decoder_bottom_hidden)
            loss += criterion(decoder_bottom_output, target[di])
            topv, topi = torch.topk(decoder_bottom_output, 1)
            decoder_prev_bottom_output = topi.squeeze(0)
            if topi.squeeze().detach() == EOS_token:
                break
    
    loss.backward()
    
    decoder_optimizer.step()
    encoder_optimizer.step()
    
    return loss.item() / target_length

def train_iters(encoder, decoder, n_epochs, print_every=10, learning_rate=0.01):
    
    print_loss_total = 0
    count = 0
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    for epoch in range(n_epochs):
        
        for idx in range(1, len(trainset) + 1):
            feats, caption = trainset[idx-1]
            
            loss = train(feats, caption, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        criterion)
            
            print_loss_total += loss
            count += 1
            
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / count
            count = 0
            print_loss_total = 0
            print(f'Epoch: {epoch} ------------- Average loss: {print_loss_avg}')
            
        if epoch % 50 == 0:
            torch.save(encoder.state_dict(), f'checkpoints/encoder_{epoch}.pth')
            torch.save(decoder.state_dict(), f'checkpoints/decoder_{epoch}.pth')
            
encoder = EncoderRNN(trainset.dict_size, MAX_LENGTH, dim_hidden=512, dim_word=512)
encoder = encoder.to(device)
decoder = DecoderRNN(trainset.dict_size, MAX_LENGTH, dim_hidden=512, dim_word=512)
decoder = decoder.to(device)

train_iters(encoder, decoder, 500, print_every=5000)
        
