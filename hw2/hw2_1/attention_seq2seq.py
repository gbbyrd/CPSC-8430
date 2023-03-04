import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from dataset import HW2_Dataset

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, cell_type='gru'):
        super(EncoderRNN, self).__init__()
        self.cell_type = cell_type
        if self.cell_type == 'gru':
            self.rnn_cell = nn.GRU
        else:
            self.rnn_cell = nn.LSTM
        
        self.dim_hidden = dim_hidden
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden)
        self.input = nn.Linear(dim_vid, dim_hidden)
        
    def forward(self, vid_feat, hidden):
        vid_feat = vid_feat.unsqueeze(0).type(torch.float32)
        input = self.input(vid_feat)
        output, hidden = self.rnn(input, hidden)
        
        return output, hidden
    
    def init_hidden(self):
        if self.cell_type == 'gru':
            return None
        else:
            return None, None
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, dim_hidden, dim_output, dropout_p=0.1, max_length=30, cell_type='gru'):
        super(AttnDecoderRNN, self).__init__()
        self.cell_type = cell_type
        if self.cell_type == 'gru':
            self.rnn_cell = nn.GRU
        else:
            self.rnn_cell = nn.LSTM
            
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(self.dim_output, self.dim_hidden)
        self.attn = nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        self.attn_combine = nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        hidden = hidden.unsqueeze(0).cuda()
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), dim=1)))
        attn_weights = attn_weights.unsqueeze(0)
        encoder_outputs = encoder_outputs.view(1, 256, -1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        print(attn_applied.size())
        print(embedded.size())
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def init_hidden(self):
        if self.cell_type == 'gru':
            return None
        else:
            return None, None

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, max_length=80,
          teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.dim_hidden, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[0]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == 1:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def train_epochs(encoder, decoder, trainset, device, epochs=100, print_every=1000, 
                 learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        
        for iter in range(1, len(trainset)+1):
            feats, caption = trainset[iter-1]

            loss = train(feats, caption, encoder, decoder, encoder_optimizer, 
                         decoder_optimizer, criterion, device)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'Epoch: {epoch+1} ----------- Average Loss: {print_loss_avg}')
            
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    trainset = HW2_Dataset(train=True)
    dim_hidden = 256
    encoder = EncoderRNN(dim_vid=4096, dim_hidden=dim_hidden, cell_type='gru')
    attn_decoder = AttnDecoderRNN(dim_hidden, trainset.dict_size, dropout_p=0.1).to(device)
    
    for name, param in encoder.named_parameters():
        print(name)
        print(param.dtype)
        print('-------------------')
    
    train_epochs(encoder, attn_decoder, trainset, device, epochs=100, print_every=1, learning_rate=.001)
        
