import torch
import numpy as np
from torch.utils.data import Dataset
import os
import json
import re

class HW2_Dataset(Dataset):
    """This dataset is used to load up feature and caption training and testing
    data from the MLDS data set. Here is the link to download the data:
    https://drive.google.com/file/d/1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ/view
    No preprocessing is necessary as the data is already in a processed format.
    Extract the data and ensure that it is inside a MLDS_hw2_1_data folder inside
    the same directory as the dataset.

    Args:
        Dataset (PyTorch Class): Build in PyTorch dataset class.
    """
    def __init__(self, data_folder=None, train=True, load_into_ram=True):
        super(HW2_Dataset, self).__init__()
        # Specify the folders for training or testing datasets
        if not data_folder:
            self.train_label_path = 'MLDS_hw2_1_data/training_label.json'
            self.train_feat_folder = 'MLDS_hw2_1_data/training_data/feat'
            self.test_label_path = 'MLDS_hw2_1_data/testing_label.json'
            self.test_feat_folder = 'MLDS_hw2_1_data/testing_data/feat'
        else:
            self.train_label_path = data_folder+'/training_label.json'
            self.train_feat_folder = data_folder+'/training_data/feat'
            self.test_label_path = data_folder+'/testing_label.json'
            self.test_feat_folder = data_folder+'/testing_data/feat'
        
        # Specify the global feat folder for the __getitem__ function
        self.feat_folder = ...
        if train:
            self.feat_folder = self.train_feat_folder
        else:
            self.feat_folder = self.test_feat_folder
        
        # Declare the beginning of sentence and end of sentence tags
        self.pad = '<PAD>'
        self.sos = '<SOS>'
        self.eos = '<EOS>'
        self.unk = '<UNK>'
        
        self.load_into_ram = load_into_ram
        
        # Load the caption and id dictionary from .json file for the training
        # and testing datasets
        self.train_label_dict, self.test_label_dict = ..., ...
        with open(self.train_label_path) as f:
            self.train_label_dict = json.load(f)
        with open(self.test_label_path) as f:
            self.test_label_dict = json.load(f)
        
        # Define the minimum amount of times a word has to be used for it to be a good
        # word and not a 'rare' word
        self.min_word_count = 3
        self.word_count = {}

        # Get the index to word and word to index dictionaries
        self.idx_to_word, self.word_to_idx = self.get_word_dictionary()
        
        # Transform the label dictionary to hold sentences where the words are represented
        # by numbers
        '''
        It is only necessary to do this for either the training or testing
        dataset, depending on what type of dataset you defined.'''
        self.label_dict = ...
        if train:
            self.label_dict = self.train_label_dict
        else:
            self.label_dict = self.test_label_dict
            
        if self.load_into_ram:
            self.ram_feats = self.load_feats()
            
        self.pairs = self.get_pairs()
        self.vocab_size = len(self.idx_to_word)
    
    def __getitem__(self, index):
        # Get all captions for the video at the index
        video_name, caption = self.pairs[index]
        
        feats = ...
        if self.load_into_ram:
            feats = self.ram_feats[video_name]
        else:
            # Load the feature into a numpy array
            feature_path = os.path.join(self.feat_folder, video_name+'.npy')
            feats = np.load(feature_path)
            
        caption = torch.tensor(caption)
        feats = torch.tensor(feats)
        feats += torch.Tensor(feats.size()).random_(0, 2000)/10000.
        feats = feats.type(torch.float32)
        '''
        Note: This will return a caption of varying length. The feature will be of the
        size 80 x 4096. You will need to manage this after the dataloader returns it.
        '''
        
        return feats, caption
        
    def __len__(self):
        return len(self.pairs)
    
    def get_word_dictionary(self):
        reserved_keys = {0:self.pad, 1:self.sos, 2:self.eos, 3:self.unk}
        count = 2
        # Count the words, split the words into rare words (which will not be included
        # in the vocabulary) and good words (which will be included)
        
        # Only do this for the training set
        for key in self.train_label_dict:
            sentences = key["caption"]
            for sentence in sentences:
                words = re.sub('[.!,;?]]', ' ', sentence).split()
                for word in words:
                    word = word.replace(".","")
                    word = word.lower()
                    self.word_count[word] = self.word_count.get(word, 0) + 1
        
        rare_words = [word for word, count in self.word_count.items() if count <= self.min_word_count]
        vocab = [word for word, count in self.word_count.items() if count > self.min_word_count]
        
        idx_to_word = {i+4: word for i, word in enumerate(vocab)}
        word_to_idx = {word: i+4 for i, word in enumerate(vocab)}
        
        for num, word in reserved_keys.items():
            idx_to_word[num] = word
            word_to_idx[word] = num
        
        return idx_to_word, word_to_idx
    
    def get_pairs(self):
        """Get full dataset in the form.

        Args:
            label_dict (dict()): dictionary that holds the video names and all captions for each video

        Returns:
            list(): ['video_feats_folder_path', 'num encoded sentence'] for all videos and captions
        """
        pairs = []
        for item in self.label_dict:
            id = item["id"]
            for caption in item["caption"]:
                transformed_caption = self.transform_caption(caption)
                pairs.append([id, transformed_caption])
        
        return pairs
    
    def transform_caption(self, sentence):
        words = sentence.split()
        transformed_caption = []
        transformed_caption.append(self.word_to_idx[self.sos])
        for word in words:
            word = word.replace(".", "")
            word = word.lower()
            if word not in self.word_to_idx:
                transformed_caption.append(self.word_to_idx[self.unk])
            else:
                transformed_caption.append(self.word_to_idx[word])
        transformed_caption.append(self.word_to_idx[self.eos])
        
        return transformed_caption
    
    def load_feats(self):
        feats = {}
        for pair in self.label_dict:
            vid_name = pair['id']
            feats_path = os.path.join(self.feat_folder, vid_name+'.npy')
            feats[vid_name] = np.load(feats_path)
        
        return feats
        
    
    def caption_to_words(self, caption):
        sentence = []
        caption = caption.numpy()
        for num in caption:
            word = self.idx_to_word[num]
            sentence.append(word)
            
        return sentence
    
    def index2sentence(self, index_seq):
        return [self.idx_to_word[int(i)] for i in index_seq]
    
if __name__ == '__main__':
    trainset = HW2_Dataset(train=True)
    
    feats, caption = trainset[0]
    