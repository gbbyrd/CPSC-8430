import torch
import json
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import random

class SquadDataset(Dataset):
    def __init__(self, train=True):
        super(SquadDataset, self).__init__()
        """This dataset loads the data into 3 synced lists:
        context, question, answer.
        
        It then creates encodings using the BertTokenizerFast tokenizer
        """
        if train:
            self.data_path = 'data/squad1.1/train-v1.1.json'
        else:
            self.data_path = 'data/squad1.1/dev-v1.1.json'
        
        # Sync the context, question, and answer data
        contexts, questions, answer_starts, answer_ends = self.read_data()
        answers = [{'answer_start': answer_starts[i], 'text': contexts[i][answer_starts[i]:answer_ends[i]], 'answer_end': answer_ends[i]} for i in range(len(answer_starts))]
        
        # Tokenize the question and contexts
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        self.encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        self.add_token_positions(answers)
        
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
                        answer_end = answer_start + len(answer_text)
                        contexts.append(context)
                        questions.append(question)
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_end)
                        
            return contexts, questions, answer_starts, answer_ends
        
    def add_token_positions(self, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(self.encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(self.encodings.char_to_token(i, answers[i]['answer_end']-1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
            
        
        