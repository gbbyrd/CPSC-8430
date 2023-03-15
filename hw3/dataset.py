import torch
import json
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import random

class SpokenSquadDataset(Dataset):
    def __init__(self, train=True):
        super(SpokenSquadDataset, self).__init__()
        """This dataset loads the data into 3 synced lists:
        context, question, answer.
        
        It then creates encodings using the BertTokenizerFast tokenizer
        """
        if train:
            self.data_path = 'data/spoken_train-v1.1.json'
        else:
            self.data_path = 'data/spoken_test-v1.1.json'
        
        # Sync the context, question, and answer data
        contexts, questions, answer_starts, answer_ends = self.read_data()
        
        # Tokenize the question and contexts
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        self.encodings = self.tokenizer(questions, contexts, truncation=True, padding=True) 
        self.encodings['answer_starts'] = answer_starts
        self.encodings['answer_ends'] = answer_ends
        
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            
    def __len__(self):
        return len(self.encodings['input_ids'])
            
    def read_data(self):
        
        with open(self.data_path) as f:
            data = json.load(f)['data']
            
            contexts = []
            questions = []
            answer_starts = []
            answer_ends = []
            
            for title in data:
                for paragraph in title['paragraphs']:
                    context = paragraph['context']
                    for qas in paragraph['qas']:
                        question = qas['question']
                        answer_text = qas['answers'][0]['text']
                        answer_start = qas['answers'][0]['answer_start']
                        
                        # add an 'answer_end' to the answer
                        num_words_in_ans = len(answer_text.split())
                        answer_end = answer_start + num_words_in_ans - 1
                        contexts.append(context)
                        questions.append(question)
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_end)
                        
            return contexts, questions, answer_starts, answer_ends