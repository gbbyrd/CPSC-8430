import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import cv2
import glob
import os    

class Seq2Seq(nn.Module):
    def __init__(self, dict_size):
        super(Seq2Seq, self).__init__()
        
        # Instantiate pre-trained resnet50 model, modify final layer, freeze all
        # other layers so that only the final layer is learned
        self.cnn = resnet18(ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(512, 500)
        for name, param in self.cnn.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            else:
                param.requires_grad = False
        
        # Instantiate lstm stack with 100 hidden units per cell
        self.lstm_stack = LSTMStack(500, 500, 1000, 500, 1, dict_size)
        self.prev_bottom_output_embedding = nn.Embedding(dict_size, 500)
        self.output_to_dictionary_dimension = nn.Linear(500, dict_size)
        self.softmax = nn.Softmax(dim=1)
        self.isEncoder = True
        self.pad = torch.zeros([1, 500])
    
    def forward(self, input, prev_bottom_output):
        lstm_top_input = ...
        lstm_prev_bottom_output = ...
        # print(input)
        if self.isEncoder:
            lstm_top_input = self.cnn(input)
            # print(lstm_top_input)
            lstm_prev_bottom_output = self.pad
        else:
            lstm_top_input = self.pad
            lstm_prev_bottom_output = self.prev_bottom_output_embedding(prev_bottom_output)
            
        # print(lstm_top_input.size(), lstm_prev_bottom_output.size())
            
        output = self.lstm_stack(lstm_top_input, lstm_prev_bottom_output)
        # print(output)
        output = self.output_to_dictionary_dimension(output)
        # print(output)
        output = self.softmax(output)
        # print(output.size())
        # print(torch.max(output))
        return output
        
class LSTMStack(nn.Module):
    def __init__(self, top_input_size, top_hidden_size, bottom_input_size, 
                 bottom_hidden_size, num_hidden_units, dict_size):
        super(LSTMStack, self).__init__()
        ''' 
        Top input = 500 dim vector
        Top output = 500 dim vector
        Bottom input = 1000 dim vector (top output concatenated with bottom output)
        Bottom output = 500 dim vector
        '''
        self.top_lstm = nn.LSTM(top_input_size, top_hidden_size, num_layers=num_hidden_units)
        self.bottom_lstm = nn.LSTM(bottom_input_size, bottom_hidden_size, num_layers=num_hidden_units)
        
    def forward(self, top_input, prev_bottom_output):
        top_output, (top_h, top_c) = self.top_lstm(top_input)
        bottom_input = torch.cat((prev_bottom_output, top_output), dim=1)
        bottom_output, (bottom_h, bottom_c) = self.bottom_lstm(bottom_input)
        
        return bottom_output
    
def decode_sentence(dictionary, sentence):
    words = []
    for number in sentence:
        words.append(dictionary[number])
        
    return words

def train_one_iteration(model, frames, caption):
    pass
    
    

def train_model(model, dataloader, epochs, device):
    
    model = seq2seq()
    model.train()
    model = model.to(device)
    
    for epoch in epochs:
        
        for frames, caption in dataloader:
            frames = frames.to(device)
            caption = caption.to(device)
            
            
    
    
        
if __name__=='__main__':
    # test dataset
    trainset = MSVD()
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
    num_to_word = trainset.num_to_word
    word_to_num = trainset.word_to_num
    seq2seq = Seq2Seq(len(num_to_word))
    for name, param in seq2seq.named_parameters():
        print(name, param.size())
    # print(seq2seq.lstm_stack.bottom_lstm.weight_hh_l97.grad())
    pad = torch.zeros([1, 500])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(seq2seq.parameters(), lr=.2)
    frames1, caption1 = ..., ...
    for frames, caption in dataloader:
        # frames1 = frames
        # caption1 = caption
        for idx, frame in enumerate(frames):
            output = seq2seq(frame, pad)
            loss = criterion(output, caption[idx])
            print(seq2seq.cnn.fc.weight.grad)
            loss.backward()
            for name, param in seq2seq.named_parameters():
                if param.grad is not None:
                    print(name, param.grad)
            optimizer.step()
            optimizer.zero_grad()
    # for epoch in range(100):
    #     for idx, frame in enumerate(frames):
    #         if idx >= len(caption1):
    #             break
    #         output = seq2seq(frame, pad)
    #         topk, topi = torch.topk(output, 1)
    #         if epoch % 5 == 0:
    #             print(topi)
    #             print(caption[idx])
    #         loss = criterion(output, caption[idx])
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
        
            
    
        
    
    
    
    
    
    
        
        
        