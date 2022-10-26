import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('utils')[0])
#sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('utils')[0]+"utils")
from nces import BaseConceptSynthesis
import numpy as np, torch, pandas as pd
from .data import Data
import random
import copy

class CSDataLoader(BaseConceptSynthesis, Data, torch.utils.data.Dataset):
    def __init__(self, data, kwargs):
        super(CSDataLoader, self).__init__(kwargs)
        self.data_triples = Data(kwargs)
        self.data_raw = data
        self.shuffle_examples = kwargs.shuffle_examples
        
        
    def load_embeddings(self, embedding_model):
        self.embeddings, _ = embedding_model.get_embeddings().cpu()
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        if self.shuffle_examples:
            random.shuffle(pos)
            random.shuffle(neg)
        #assert '#' in pos[0] or '.' in pos[0], 'Namespace error, expected separator # or .'
        #print("Emb Mat: ", self.embeddings.shape)
        #print("\nPos\n")
        #print(self.data_triples.entity2idx.loc[pos].values)
        datapoint_pos = self.embeddings[self.data_triples.entity2idx.loc[pos].values.squeeze()]
        datapoint_neg = self.embeddings[self.data_triples.entity2idx.loc[neg].values.squeeze()]
        labels, length = self.get_labels(key)
        #torch.cat([datapoint_pos, datapoint_neg], 0)
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(self.max_length-length)]).long()
            
        
        
class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec