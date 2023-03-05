#!/usr/bin/env python3

""" 
    This is the model for hw2 of the CPSC 8430 - Deep Learning class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit

class attention(nn.Module):
    def __init__(self, dim_hidden):
        super(attention, self).__init__()
        
        self.dim_hidden = dim_hidden
        self.linear1 = nn.Linear(2*dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, dim_hidden)
        self.linear4 = nn.Linear(dim_hidden, dim_hidden)
        self.to_weight = nn.Linear(dim_hidden, 1, bias=False)
        
    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.dim_hidden)
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attn_weights = self.to_weight(x)
        attn_weights = attn_weights.view(batch_size, seq_len)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, dropout_p=0.25):
        super(EncoderRNN, self).__init__()
        
        # class parameters
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.dropout_p = dropout_p
        
        # layers
        self.linear_in = nn.Linear(in_features=dim_vid, out_features=dim_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(dim_hidden, dim_hidden, batch_first=True)
        
    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.linear_in(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.dim_hidden)
        
        output, hidden = self.rnn(input)
        
        return output, hidden
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, dim_hidden, dim_output, dim_vocab, dim_word, dropout_p=.25):
        super(AttnDecoderRNN, self).__init__()
        
        # class parameters
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_vocab = dim_vocab
        self.dim_word = dim_word
        self.dropout_p = dropout_p
        
        # layers
        self.embedding = nn.Embedding(dim_vocab, dim_word)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.GRU(dim_hidden+dim_word, dim_hidden, batch_first=True)
        self.attn = attention(dim_hidden)
        self.linear_out = nn.Linear(dim_hidden, dim_output)
        
    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.init_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = torch.ones(batch_size, 1).long()
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        
        # for scheduled sampling
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()
        
        for i in range(seq_len-1):
            threshold = self.get_teacher_learning_ratio(training_steps=tr_steps)
            current_input_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold \
                else self.embedding(decoder_current_input_word).squeeze(1)
                
            # get contect from attention to input into cell
            context = self.attn(decoder_current_hidden_state, encoder_output)
            rnn_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            rnn_output, decoder_current_hidden_state = self.rnn(rnn_input, decoder_current_hidden_state)
            logprob = self.linear_out(rnn_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]
            
        # concatenate all of the predicted probabilities in the dim (batch, seq_length, output_size)
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
    
    def inference(self, encoder_last_hidden_state, encoder_output):
        ''' Does not use teacher forcing'''
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.init_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = torch.ones(batch_size, 1).long()
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        assumed_seq_len = 28
        
        for i in range(assumed_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            
            # get contect from attention to input into cell
            context = self.attn(decoder_current_hidden_state, encoder_output)
            rnn_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            rnn_output, decoder_current_hidden_state = self.rnn(rnn_input, decoder_current_hidden_state)
            logprob = self.linear_out(rnn_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]
            
        # concatenate all of the predicted probabilities in the dim (batch, seq_length, output_size)
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
            
    
    def init_hidden_state(self, last_encoder_hidden_state):
        if last_encoder_hidden_state is None:
            return None
        else:
            return last_encoder_hidden_state
        
    def get_teacher_learning_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, vid_feats, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(vid_feats)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state, encoder_outputs,
                                                        target_sentences, mode, tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.inference(encoder_last_hidden_state, encoder_outputs)
        
        return seq_logProb, seq_predictions