import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import math
import random
from utils import *

class CustomDataset(Dataset):
    def __init__(self,char_data,strokes):
        self.char_data = char_data
        self.strokes = strokes
    def __len__(self):
        assert(len(self.char_data) == len(self.strokes))
        return len(self.char_data)
    def __getitem__(self,i):
        return self.char_data[i], self.strokes[i]


def fix_length(data,max_len,pad_idx):
    if pad_idx:
        new_data = np.zeros((len(data),max_len,3))
    else:
        new_data = np.zeros((len(data),max_len))

    for i,d in enumerate(data):
        diff = max_len - len(d)
        if diff != 0:
            pad = [pad_idx]*diff
            if pad_idx:
                d = (np.r_[d,pad]).astype(np.float32)
            else:
                d.extend(pad)
        new_data[i] = d
    return new_data


def custom_collate(batch):
    char_data = [item[0] for item in batch]
    strokes = [item[1] for item in batch]
    max_char = max([len(c) for c in char_data])
    char_data = fix_length(char_data,max_char,pad_idx=0)
    max_stroke = max([len(s) for s in strokes])
    strokes = fix_length(strokes,max_stroke,pad_idx=[4,0,0])
    return [torch.as_tensor(char_data,dtype=torch.long),torch.as_tensor(strokes,dtype=torch.float32)]


START = [[2,0,0]]
STOP = [[3,0,0]]
PAD = [[4,0,0]]

class ProcessStrokes():

    def __init__(self):
        self.means = 0
        self.stds = 0

    def normalize(self,strokes):
        """ Mean and std for relative coords """
        row = np.array([bit for s in strokes for bit in s[:,1]])
        col = np.array([bit for s in strokes for bit in s[:,2]])
        data_mean = np.array([row.mean(), col.mean()])
        data_std = np.array([row.std(), col.std()])

        norm_strokes = []
        for s in strokes:
            s[:,1:] = (s[:,1:] - data_mean)/(data_std)
            norm_strokes.append(s)
        
        self.means = data_mean
        self.stds = data_std

        return norm_strokes

    def add_tokens(self,strokes):
        """" Add start and stop tokens """
        for i in range(len(strokes)):
            strokes[i] = (np.r_[START,strokes[i],STOP]).astype(np.float32)
        return strokes

    def denormalize(self,stroke,remove_tokens =True):
        """ Remove start stop and pad tokens and denormalize """
        if remove_tokens:
            where_pad = np.where(stroke[:,0] == 4)[0]
            if where_pad.size > 0:
                start_pad = where_pad[0]
                stroke = stroke[:start_pad] # remove pad
            
            where_end = np.where(stroke[:,0] == 3)[0] # remove stop
            if where_end.size > 0:
                stroke = stroke[:where_end[0]]
            
            where_start = np.where(stroke[:,0]==2)[0] # remove start
            if where_start.size > 0:
                stroke = stroke[where_start[0]:]
       
        # denormalize
        stroke[:,1:] =  (stroke[:,1:] * self.stds) + self.means
        return stroke


class LossMetric():
    """Calculate Loss for the entire epoch"""
    def __init__(self,mask_end=True):
        self.loss_ce = nn.CrossEntropyLoss(reduction='none')
        self.total_celoss = 0
        self.total_nllloss= 0
        self.total_samples = 0
        self.mask_end = mask_end

    def mask_loss(self,preds,y_true):
        
        intrp,coords = y_true[:,:,0].long(), y_true[:,:,1:]
        
        if self.mask_end:
            mask = (~((intrp==2)|(intrp==3)|(intrp==4))).float()
        else:
             mask = (~(intrp==4)).float()

        ce_preds,pi = preds[:2]
        ce_preds = ce_preds.transpose(-1,1)
        ce = self.loss_ce(ce_preds, intrp)
        ce = mask*ce

        probs = pi*calc_prob(coords,* preds[2:]) 
        probs = torch.sum(probs,-1)
        nll = -mask*(torch.log(probs + 1e-10))

        return ce.sum().item(), nll.sum().item(), mask.sum().item()

    def update_state(self,preds, y_true):
        ce, nll, num = self.mask_loss(preds,y_true)
        self.total_nllloss += nll
        self.total_celoss += ce
        self.total_samples += num

    def result(self):
        ce = self.total_celoss/ self.total_samples
        nll = self.total_nllloss/ self.total_samples
        return ce, nll,  ce+ nll

    def reset_state(self):
        self.total_celoss = 0
        self.total_nllloss= 0
        self.total_samples = 0


def batch_loader(strokes,batch_sz,bptt,buffer=2,debug_state=False):
    """Yields batches with psuedo shuffling and padding"""

    # sort strokes according to len
    #pdb.set_trace()
    if not debug_state:
        strokes = sorted(strokes,key = lambda x: len(x))
    tot_bz = math.ceil(len(strokes)/batch_sz)
    # collect buckets of 'batch_sz*buffer' inputs
    all_buckets = []
    for i in range(tot_bz):
        batch = strokes[i*batch_sz*buffer:(i+1)*batch_sz*buffer]
        all_buckets.append(batch)

    def shuffle(x,return_x=False,debug_state=False):
        #pdb.set_trace()
        if not debug_state:
            random.shuffle(x)
        if return_x:
            return x

    # shuffle order of buckets
    for bucket in shuffle(all_buckets,True,debug_state):
        # shuffle a particular bucket
        shuffle(bucket,False,debug_state)
        for k in range(math.ceil(len(bucket)/batch_sz)):
            # get a batch from the bucket
            batch = bucket[k*batch_sz:(k+1)*batch_sz]
            # set seq_len for batch and pad if needed
            seq_len = max([len(s) for s in batch])
            for j in range(len(batch)):
                diff = seq_len - len(batch[j])
                if diff != 0:
                    #pdb.set_trace()
                    pad = np.array(PAD*diff)
                    batch[j] = (np.r_[batch[j],pad]).astype(np.float32)
            
            batch = np.array(batch)
            total_seq = (seq_len-1)//bptt
            new_set = True # track if same set or new
            # yield bptt len batch
            for k in range(total_seq):
                x = batch[:,k*bptt:(k+1)*bptt]
                y = batch[:,k*bptt+1:(k+1)*bptt+1]
                yield (new_set,x,y)
                new_set = False 
            
            if (seq_len-1)%bptt != 0: # any remaining seq
                x = batch[:,(k+1)*bptt:-1]
                y = batch[:,(k+1)*bptt+1:]
                yield (new_set,x,y)


class ProcessText():

    def __init__(self):
        self.char_data= []
        self.word2idx = {}
        self.idx2word = {}
    
    def tokenize(self,texts):
        char_list = []
        for t in texts:
            char_list.append([c for c in t.strip()])
        for c in char_list:
            c.insert(0,'<start>')
            c.append('<end>')
        self.char_list = char_list
        return char_list

    def create_vocab(self):
        flat = []
        for t in self.char_list:
            for c in t:
                flat.append(c)
        unq_char,counts = np.unique(flat,return_counts=True)

        pad = '<pad>'
        word2idx = {word: i+1 for i,word in enumerate(unq_char)}
        word2idx[pad] = 0
        idx2word = {i+1: word for i,word in enumerate(unq_char)}
        idx2word[0]=pad
        self.word2idx, self.idx2word = word2idx, idx2word
        return word2idx, idx2word

    def numericalize(self):
        char_data = []
        for c in self.char_list:
            num_sent = []
            for token in c:
                num_sent.append(self.word2idx[token])
            char_data.append(num_sent)
        self.char_data = char_data
        return char_data
    
    def sent2num(self,sent):
        tokens = [c for c in sent.strip()]
        tokens.insert(0,'<start>')
        tokens.append('<end>')
        num_sent = [self.word2idx[t] for t in tokens]
        return num_sent
    
    def num2sent(self,nums):
        sent =[]
        for i in nums:
            if self.idx2word[i] == '<start>' or self.idx2word[i] == '<end>' or self.idx2word[i] == '<pad>': 
                continue
            sent.append(self.idx2word[i])

        return ''.join(sent)




def collect_one_batch(data,batch_loader):
    """Sanity check for batch_loader """
    sets,cat_arr = 0, []
    for is_set,x,y in batch_loader(data, 10, 100, 1 ,True):
        if is_set == True:
            sets +=1
        if sets == 2:
            break
        cat_arr.append(x)
    return np.concatenate(cat_arr, axis=1)

    
        















