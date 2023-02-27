import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import cv2
import glob
import os

# class LangDictionary():
#     def __init__(self, list_of_words):
        
#         self.n_words = 2 # bos and eos tokens
#         self.n_words += len(list_of_words)
        
#         self.word_dict = self.get_word_dict(list_of_words)
        
#     def get_word_dict(self, list_of_words):
#         word_dict = dict()
#         word_dict[0] = 'bos'
#         word_dict[1] = 'eos'
#         for idx, word in enumerate(list_of_words):
#             word_dict[idx+2] = word 
            
#         return word_dict

class MSVD(Dataset):
    def __init__(self):
        super(MSVD, self).__init__()
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.captions, self.video_names, self.num_to_word, self.word_to_num = self.get_captions_video_names_word_dict()
        
    def __getitem__(self, index):
        # Get the list of image tensors for each frame in the video
        frames = glob.glob(os.path.join(self.video_names[index], '*'))
        frames.sort()
        frames_tensor = []
        for frame in frames:
            image = cv2.imread(frame)
            image = self.transform(image)
            frames_tensor.append(image)
            
        caption = self.captions[index]
        
        return frames_tensor, caption
    
    def __len__(self):
        return len(self.captions)
    
    def get_captions_video_names_word_dict(self):
        captions = []
        video_names = []
        clips_folder = 'data/clips'
        with open('data/captions/captions.txt', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                sentence = (line.split(" ", 1)[1])
                captions.append(sentence.split())
                video_names.append(os.path.join(clips_folder, line.split(" ", 1)[0]))
                
        num_to_word, word_to_num = self.get_dictionary(captions)
        self.word_to_num = word_to_num
        captions = self.get_one_hot_captions(captions)
        
        return captions, video_names, num_to_word, word_to_num
    
    def get_dictionary(self, captions):
        word_set = set()
        list_of_words = []
        
        for sentence in captions:
            for word in sentence:
                if word in word_set:
                    continue
                word_set.add(word)
                list_of_words.append(word)
            
        word_to_num = dict()
        num_to_word = dict()
        num_to_word[0] = 'bos'
        num_to_word[1] = 'eos'
        word_to_num['bos'] = 0
        word_to_num['eos'] = 1
        for idx, word in enumerate(list_of_words):
            num_to_word[idx+2] = word 
            word_to_num[word] = idx+2
            
        return num_to_word, word_to_num
    
    def get_one_hot_captions(self, captions):
        one_hot_captions = []
        for sentence in captions:
            one_hot_sentence = []
            for word in sentence:
                one_hot_sentence.append(self.word_to_num[word])
            one_hot_captions.append(one_hot_sentence)
            
        # # Change every list of one_hot_captions to a tensor
        # for idx, caption in enumerate(one_hot_captions):
        #     one_hot_captions[idx] = torch.tensor(caption)
        
        return one_hot_captions      

class Seq2Seq(nn.Module):
    def __init__(self, dict_size):
        super(Seq2Seq, self).__init__()
        
        # Instantiate pre-trained resnet50 model, modify final layer, freeze all
        # other layers so that only the final layer is learned
        self.cnn = resnet50(ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(2048, 500)
        for name, param in self.cnn.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            else:
                param.requires_grad = False
        
        # Instantiate lstm stack with 100 hidden units per cell
        self.lstm_stack = LSTMStack(500, 500, 1000, 500, 100, dict_size)
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
    pad = torch.zeros([1, 500])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(seq2seq.parameters(), lr=.2)
    frames1, caption1 = ..., ...
    for frames, caption in dataloader:
        frames1 = frames
        caption1 = caption
        break
        
        # for idx, frame in enumerate(frames):
        #     output = seq2seq(frame, pad)
        #     loss = criterion(output, caption[idx])
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
    for epoch in range(100):
        for idx, frame in enumerate(frames):
            if idx >= len(caption1):
                break
            output = seq2seq(frame, pad)
            topk, topi = torch.topk(output, 1)
            if epoch % 5 == 0:
                print(topi)
                print(caption[idx])
            loss = criterion(output, caption[idx])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            
    
        
    
    
    
    
    
    
        
        
        