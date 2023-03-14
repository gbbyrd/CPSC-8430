import requests
import json
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForQuestionAnswering

from dataset import SquadDataset

"""Finetune the BERT model for question answering

"""

trainset = SquadDataset(train=True)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

for batch in trainloader:
    print(batch)
    