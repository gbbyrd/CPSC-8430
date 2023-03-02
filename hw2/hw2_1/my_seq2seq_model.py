import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import cv2
import glob
import os    

from dataset import HW2_Dataset

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=4096, bos_id=0, 
                 eos_id=1, n_layers=1):
        super(Seq2Seq, self).__init__()
        self.rnn_cell = nn.LSTM
        
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers, batch_first=True)
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers, batch_first=True)

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        
    def forward(self, vid_feats, target_variable=None):
        # Only try with one frame to start
        
        padding_vid = torch.zeros([1, 4096])
        padding_word = torch.zeros([1, self.dim_word])
        state1 = None
        state2 = None
        
        # First pass through top lstm
        output1, state1 = self.rnn1(vid_feats, state1)
        
        # Concatenate the output of the first lstm with the padded word of dimension=dim_word
        input2 = torch.cat((output1, padding_word), dim=1)
        
        # First pass through the bottom lstm
        output2, state2 = self.rnn2(input2, state2)
        output2 = torch.tensor(self.bos_id)
        prob_sentence = []
        for i in range(self.max_length-1):
            # Send both lstm models to a contiguous chunk of memory for memory optimization
            self.rnn1.flatten_parameters()
            self.rnn2.flatten_parameters()
            output1, state1 = self.rnn1(padding_vid, state1)
            embedded_word = self.embedding(output2).view(1, -1)
            input2 = torch.cat((output1, embedded_word), dim=1)
            output2, state2 = self.rnn2(input2, state2)
            logits = self.out(output2)
            logits = F.log_softmax(logits)
            prob_sentence.append(logits)
            topv, output2 = torch.topk(output2, 1)
            # print(output2)
        return prob_sentence
    
def one_hot(x, vocab_size):
    one_hot = torch.zeros(1, vocab_size)
    one_hot[0][x] = 1
    
    return one_hot
            
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Instantiate dataset and dataloader
    trainset = HW2_Dataset(train=True)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    
    vocab_size = trainset.dict_size
    
    # Instantiate model
    model = Seq2Seq(vocab_size, max_len=20, dim_hidden=512, dim_word=512, n_layers=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    epochs = 200
    for epoch in range(epochs):
        loss = 0
        for i in range(3):
            feat, caption = trainset[i]
            feat = feat.squeeze()
            feat = feat[0].view(1, -1)
            feat = feat.to(torch.float32)
            sentence = [0, 0, 0]
            sentence[i] = trainset.caption_to_words(caption)
            prob_sentence = model(feat)
            for idx in range(len(prob_sentence)):
                one_hot_word = one_hot(caption[idx], vocab_size)
                loss += criterion(prob_sentence[idx], one_hot_word)
                
                if caption[idx].item() == 1:
                    break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss/3)
            
    def caption_to_words(caption, trainset):
        sentence = []
        for num in caption:
            word = trainset.idx_to_word[num.item()]
            sentence.append(word)
            
        return sentence
    for i in range(5, 8):
        feat, caption = trainset[i]
        feat = feat.squeeze()
        feat = feat[0].view(1, -1)
        feat = feat.to(torch.float32)
        sentence = trainset.caption_to_words(caption)
        prob_sentence = model(feat)
        pred_sentence = []
        for tensor in prob_sentence:
            topv, word_idx = torch.topk(tensor, 1)
            pred_sentence.append(word_idx)
        # print(pred_sentence)
        word_sentence = caption_to_words(pred_sentence, trainset)
        print(sentence)
        print(word_sentence)


    
        












































































# class Seq2Seq(nn.Module):
#     def __init__(self, dict_size):
#         super(Seq2Seq, self).__init__()
        
#         # Instantiate pre-trained resnet50 model, modify final layer, freeze all
#         # other layers so that only the final layer is learned
#         self.cnn = resnet18(ResNet18_Weights.DEFAULT)
#         self.cnn.fc = nn.Linear(512, 500)
#         for name, param in self.cnn.named_parameters():
#             if name == 'fc.weight' or name == 'fc.bias':
#                 continue
#             else:
#                 param.requires_grad = False
        
#         # Instantiate lstm stack with 100 hidden units per cell
#         self.lstm_stack = LSTMStack(500, 500, 1000, 500, 1, dict_size)
#         self.prev_bottom_output_embedding = nn.Embedding(dict_size, 500)
#         self.output_to_dictionary_dimension = nn.Linear(500, dict_size)
#         self.softmax = nn.Softmax(dim=1)
#         self.isEncoder = True
#         self.pad = torch.zeros([1, 500])
    
#     def forward(self, input, prev_bottom_output):
#         lstm_top_input = ...
#         lstm_prev_bottom_output = ...
#         # print(input)
#         if self.isEncoder:
#             lstm_top_input = self.cnn(input)
#             # print(lstm_top_input)
#             lstm_prev_bottom_output = self.pad
#         else:
#             lstm_top_input = self.pad
#             lstm_prev_bottom_output = self.prev_bottom_output_embedding(prev_bottom_output)
            
#         # print(lstm_top_input.size(), lstm_prev_bottom_output.size())
            
#         output = self.lstm_stack(lstm_top_input, lstm_prev_bottom_output)
#         # print(output)
#         output = self.output_to_dictionary_dimension(output)
#         # print(output)
#         output = self.softmax(output)
#         # print(output.size())
#         # print(torch.max(output))
#         return output
        
# class LSTMStack(nn.Module):
#     def __init__(self, top_input_size, top_hidden_size, bottom_input_size, 
#                  bottom_hidden_size, num_hidden_units, dict_size):
#         super(LSTMStack, self).__init__()
#         ''' 
#         Top input = 500 dim vector
#         Top output = 500 dim vector
#         Bottom input = 1000 dim vector (top output concatenated with bottom output)
#         Bottom output = 500 dim vector
#         '''
#         self.top_lstm = nn.LSTM(top_input_size, top_hidden_size, num_layers=num_hidden_units)
#         self.bottom_lstm = nn.LSTM(bottom_input_size, bottom_hidden_size, num_layers=num_hidden_units)
        
#     def forward(self, top_input, prev_bottom_output):
#         top_output, (top_h, top_c) = self.top_lstm(top_input)
#         bottom_input = torch.cat((prev_bottom_output, top_output), dim=1)
#         bottom_output, (bottom_h, bottom_c) = self.bottom_lstm(bottom_input)
        
#         return bottom_output
    
# def decode_sentence(dictionary, sentence):
#     words = []
#     for number in sentence:
#         words.append(dictionary[number])
        
#     return words

# def train_one_iteration(model, frames, caption):
#     pass
    
    

# def train_model(model, dataloader, epochs, device):
    
#     model = seq2seq()
#     model.train()
#     model = model.to(device)
    
#     for epoch in epochs:
        
#         for frames, caption in dataloader:
#             frames = frames.to(device)
#             caption = caption.to(device)
            
            
    
    
        
# if __name__=='__main__':
#     # test dataset
#     trainset = MSVD()
#     dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
#     num_to_word = trainset.num_to_word
#     word_to_num = trainset.word_to_num
#     seq2seq = Seq2Seq(len(num_to_word))
#     for name, param in seq2seq.named_parameters():
#         print(name, param.size())
#     # print(seq2seq.lstm_stack.bottom_lstm.weight_hh_l97.grad())
#     pad = torch.zeros([1, 500])
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(seq2seq.parameters(), lr=.2)
#     frames1, caption1 = ..., ...
#     for frames, caption in dataloader:
#         # frames1 = frames
#         # caption1 = caption
#         for idx, frame in enumerate(frames):
#             output = seq2seq(frame, pad)
#             loss = criterion(output, caption[idx])
#             print(seq2seq.cnn.fc.weight.grad)
#             loss.backward()
#             for name, param in seq2seq.named_parameters():
#                 if param.grad is not None:
#                     print(name, param.grad)
#             optimizer.step()
#             optimizer.zero_grad()
#     # for epoch in range(100):
#     #     for idx, frame in enumerate(frames):
#     #         if idx >= len(caption1):
#     #             break
#     #         output = seq2seq(frame, pad)
#     #         topk, topi = torch.topk(output, 1)
#     #         if epoch % 5 == 0:
#     #             print(topi)
#     #             print(caption[idx])
#     #         loss = criterion(output, caption[idx])
#     #         loss.backward()
#     #         optimizer.step()
#     #         optimizer.zero_grad()
        
            
    
        
    
    
    
    
    
    
        
        
        