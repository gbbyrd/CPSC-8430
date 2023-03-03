import torch
import numpy as np
from torch.utils.data import Dataset
import os
import json

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
    def __init__(self, train=True):
        super(HW2_Dataset, self).__init__()
        # Specify the folders for training or testing datasets
        self.train_label_path = 'MLDS_hw2_1_data/training_label.json'
        self.train_feat_folder = 'MLDS_hw2_1_data/training_data/feat'
        self.test_label_path = 'MLDS_hw2_1_data/testing_label.json'
        self.test_feat_folder = 'MLDS_hw2_1_data/testing_data/feat'
        
        # Specify the global feat folder for the __getitem__ function
        self.feat_folder = ...
        if train:
            self.feat_folder = self.train_feat_folder
        else:
            self.feat_folder = self.test_feat_folder
        
        # Declare the beginning of sentence and end of sentence tags
        self.bos = 0
        self.eos = 1
        
        # Load the caption and id dictionary from .json file for the training
        # and testing datasets
        self.train_label_dict, self.test_label_dict = ..., ...
        with open(self.train_label_path) as f:
            self.train_label_dict = json.load(f)
        with open(self.test_label_path) as f:
            self.test_label_dict = json.load(f)

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
            
        self.transformed_label_dict = self.transform_captions(self.label_dict)
        self.dict_size = len(self.idx_to_word)
    
    def __getitem__(self, index):
        # Get all captions for the video at the index
        captions = self.transformed_label_dict[index]["caption"]
        
        # There are multiple correct captions per video, choose a random one
        rand_int = np.random.randint(0, len(captions))
        # caption = captions[rand_int]
        caption = captions[0]
        feature_name = self.transformed_label_dict[index]["id"] + '.npy'
        feature_path = os.path.join(self.feat_folder, feature_name)
        
        # Load the feature into a numpy array
        feat = np.load(feature_path)
            
        caption = torch.tensor(caption).view(-1, 1)
        feat = torch.tensor(feat)
        
        '''
        Note: This will return a caption of varying length. The feature will be of the
        size 80 x 4096. You will need to manage this after the dataloader returns it.
        '''
        
        return feat, caption
        
    def __len__(self):
        return len(self.transformed_label_dict)
    
    def get_word_dictionary(self):
        idx_to_word = {0:self.bos, 1:self.eos}
        word_to_idx = {self.bos:0, self.eos:1}
        count = 2
        for key in self.train_label_dict:
            sentences = key["caption"]
            for sentence in sentences:
                words = sentence.split()
                for word in words:
                    word = word.replace(".","")
                    word = word.lower()
                    if word not in word_to_idx:
                        word_to_idx[word] = count
                        idx_to_word[count] = word
                        count += 1
        
        for key in self.test_label_dict:
            sentences = key["caption"]
            for sentence in sentences:
                words = sentence.split()
                for word in words:
                    word = word.replace(".","")
                    word = word.lower()
                    if word not in word_to_idx:
                        word_to_idx[word] = count
                        idx_to_word[count] = word
                        count += 1
        
        return idx_to_word, word_to_idx
    
    def transform_captions(self, label_dict):
        transformed_label_dict = {}
        for idx, key in enumerate(label_dict):
            transformed_captions = []
            id = key["id"]
            for caption in key["caption"]:
                transformed_caption = self.transform_caption(caption)
                transformed_captions.append(transformed_caption)
            transformed_label_dict[idx] = {}
            transformed_label_dict[idx]["caption"] = transformed_captions
            transformed_label_dict[idx]["id"] = id
        
        return transformed_label_dict
    
    def transform_caption(self, sentence):
        words = sentence.split()
        transformed_caption = []
        for word in words:
            word = word.replace(".", "")
            word = word.lower()
            transformed_caption.append(self.word_to_idx[word])
        transformed_caption.append(self.eos)
        
        return transformed_caption
    
    def caption_to_words(self, caption):
        sentence = []
        caption = caption.numpy()
        for num in caption:
            word = self.idx_to_word[num]
            sentence.append(word)
            
        return sentence
    