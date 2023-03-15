import requests
import json
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW

from dataset import SpokenSquadDataset

"""Finetune the BERT model for question answering

"""

trainset = SpokenSquadDataset(train=True)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

print(f'Working on {device}')

N_EPOCHS = 5
optim = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

for epoch in range(N_EPOCHS):
    loop = tqdm(trainloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        answer_starts = batch['answer_starts'].to(device)
        answer_ends = batch['answer_ends'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, 
                        start_positions=answer_starts, end_positions=answer_ends)
        loss = outputs[0]
        loss.backward()
        optim.step()
        
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())
        
model_path = 'checkpoints/'
model.save_pretrained(model_path)
SpokenSquadDataset.tokenizer.save_pretrained(model_path)