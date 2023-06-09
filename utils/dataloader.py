import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('utils')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('utils')[0]+'utils/')
from nces import BaseConceptSynthesis
import numpy as np, torch, pandas as pd
import random
import copy

class NCESDataLoader(BaseConceptSynthesis, torch.utils.data.Dataset):
    def __init__(self, data, triples_data, vocab, inv_vocab, kwargs):
        super(NCESDataLoader, self).__init__(vocab, inv_vocab, kwargs)
        self.triples_data = triples_data
        self.data_raw = data
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[pos].values.squeeze()]
        datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[neg].values.squeeze()]
        labels, length = self.get_labels(key)
        return datapoint_pos, datapoint_neg, torch.cat([torch.tensor(labels), self.vocab['PAD']*torch.ones(self.max_length-length)]).long()
    
    
    
class NCESDataLoaderInference(torch.utils.data.Dataset):
    
    """This class is similar to NCESDataLoader except that labels (class expression strings) are not needed here. This is useful for learning problems whose atoms are not present in the trained models. Still NCES instances are still able to synthesize quality solutions as they do not rely on labels."""
    
    def __init__(self, data, triples_data):
        super().__init__()
        self.triples_data = triples_data
        self.data_raw = data
        
        
    def load_embeddings(self, embedding_model):
        embeddings, _ = embedding_model.get_embeddings()
        self.embeddings = embeddings.cpu()
        

    def __len__(self):
        return len(self.data_raw)
    
    def __getitem__(self, idx):
        key, value = self.data_raw[idx]
        pos = value['positive examples']
        neg = value['negative examples']
        datapoint_pos = self.embeddings[self.triples_data.entity2idx.loc[pos].values.squeeze()]
        datapoint_neg = self.embeddings[self.triples_data.entity2idx.loc[neg].values.squeeze()]
        return datapoint_pos, datapoint_neg
            
        
        
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
    