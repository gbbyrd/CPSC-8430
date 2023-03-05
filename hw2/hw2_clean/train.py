import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import random
from my_seq2seq_model import Seq2Seq, EncoderRNN, AttnDecoderRNN
import datetime
import sys
import os
import time
from dataset import HW2_Dataset

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = None

    def forward(self, x, y, lengths):
        # first dim of x and y is the same (equals to batch size)
        batch_size = len(x)
        predict_cat = None
        groundT_cat = None
        flag = True

        for batch in range(batch_size):
            predict      = x[batch]
            ground_truth = y[batch]
            seq_len = lengths[batch] -1

            predict = predict[:seq_len]
            ground_truth = ground_truth[:seq_len]
            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False
            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

        try:
            assert len(predict_cat) == len(groundT_cat)

        except AssertionError as error:
            print('prediction length is not same as ground truth length')
            print('prediction length: {}, ground truth length: {}'.format(len(predict_cat), len(groundT_cat)))

        self.loss = self.loss_fn(predict_cat, groundT_cat)
        self.avg_loss = self.loss/batch_size
        
        return self.loss
    
# Create mini-batch tensors from the list of tuples (image, caption)
def minibatch(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths
    
class training(object):
    def __init__(self, model, train_dataloader=None, test_dataloader=None, helper=None):
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        # Use cuda is available
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
            print('GPU is available')
        else:
            self.model = model.cpu()

        # define hyper parameters
        self.parameters = model.parameters()
        self.loss_fn = LossFun()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=0.001)
        self.helper = helper

    def train(self, epoch):
        self.model.train()

        test_avi, test_truth = None, None

        for batch_idx, batch in enumerate(self.train_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # start training process
            self.optimizer.zero_grad()
            seq_logProb, seq_predictions = self.model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
            # print(seq_logProb.size())
            # eliminate <SOS>
            ground_truths = ground_truths[:, 1:]  
            loss = self.loss_fn(seq_logProb, ground_truths, lengths)
            loss.backward()
            self.optimizer.step()

            # print out training info
            if (batch_idx+1):
                info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader), loss=loss.item())
                print(info, end='\r')
                sys.stdout.write("\033[K")

        info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader), loss=loss.item())
        print(info)
        
        # update loss for each epoch
        self.loss = loss.item()


    def eval(self):
        # set model to evaluation(testing) mode
        self.model.eval()
        test_predictions, test_truth = None, None
        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            ground_truths = ground_truths[:, 1:]
            test_predictions = seq_predictions[:3]
            test_truth = ground_truths[:3]
            break


    def test(self):
        
        # set model to evaluation(testing) mode
        self.model.eval()
        ss = []
        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            id, avi_feats = batch
            if self.__CUDA__:
                avi_feats = avi_feats.cuda()
            id, avi_feats = id, Variable(avi_feats).float()

            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            test_predictions = seq_predictions
            result = [[x if x != '<UNK>' else 'something' for x in self.helper.index2sentence(s)] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]
            rr = zip(id, result)
            for r in rr:
                ss.append(r)
        return ss

    def get_training_info(self,**kwargs):
        ep = kwargs.pop("epoch", None)
        bID = kwargs.pop("batch_id", None)
        bs = kwargs.pop("batch_size", None)
        tds = kwargs.pop("total_data_size", None)
        nb = kwargs.pop("n_batch", None)
        loss = kwargs.pop("loss", None)
        info = "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(ep, (bID+1)*bs, tds, 100.*bID/nb, loss)
        return info
    
def main():
    # import data
    # import training and testing labels from associated json files
    print(os.getcwd())

    trainset = HW2_Dataset(train=True, load_into_ram=True)
    testset = HW2_Dataset(train=False, load_into_ram=True)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    
    dim_word = 1024
    inputFeatDim = 4096
    dim_hidden = 512
    dropout_p = 0.3
    dim_output = trainset.vocab_size
    
    epochs_n = 100
    ModelSaveLoc = 'checkpoints'
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    encoder = EncoderRNN(dim_vid=inputFeatDim, dim_hidden=dim_hidden, dropout_p=dropout_p)
    decoder = AttnDecoderRNN(dim_hidden=dim_hidden, dim_output=dim_output, dim_vocab=dim_output, dim_word=dim_word, dropout_p=dropout_p)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)
    train = training(model=seq2seq, train_dataloader=trainloader, test_dataloader=testloader)

    start = time.time()
    for epoch in range(epochs_n):
        train.train(epoch+1)
        train.eval()

    end = time.time()
    torch.save(seq2seq, "{}/{}.h5".format(ModelSaveLoc, 'model0'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))
        
if __name__ == '__main__':
    main()
