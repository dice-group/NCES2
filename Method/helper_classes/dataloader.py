import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0])
from base.base_concept_synthetizer import BaseConceptSynthesis
from functools import singledispatchmethod
import numpy as np, random, torch, pandas as pd
from typing import List
from util.data import Data
from abc import ABCMeta

random.seed(1)

class CSDataLoader(BaseConceptSynthesis, Data, metaclass=ABCMeta):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    @singledispatchmethod    
    def load(self, embeddings, data, batch_size, shuffle, **kwargs):
        raise NotImplementedError
     
    @load.register  
    def _(self, embeddings: pd.DataFrame, data=None, batch_size=128, shuffle=True, **kwargs):
        if shuffle:
            random.shuffle(data)
        assert isinstance(data, List) or isinstance(data, np.ndarray), "Expected data type List or array, got object of type {}".format(type(data))
        datapoints = []
        targets = []
        for key, value in data:
            pos = value['positive examples']
            try:
                datapoint_pos = torch.FloatTensor(list(map(lambda x: embeddings.loc[x], pos)))
            except KeyError:
                try:
                    datapoint_pos = torch.FloatTensor(list(map(lambda x: embeddings.loc[x.replace("#", ".")], pos)))
                except KeyError:
                    continue
            datapoints.append(datapoint_pos.unsqueeze(0))
            targets.append(key)
        targets_numerical = torch.zeros((len(targets), len(self.vocab), self.max_num_atom_repeat))
        for j in range(len(targets)):
            targets_numerical[j] = self.get_scores_of_atom_indices(targets[j])
        return torch.cat(datapoints), targets_numerical, np.array(targets, dtype=object)
            
    @load.register
    def _(self, embeddings: tuple, data=None, shuffle=True, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if shuffle:
            random.shuffle(data)
        for i in range(0, len(data)-batch_size+1, batch_size):
            datapoints = []
            targets = []
            data_ = data[i:i+batch_size]
            for key, value in data_:
                pos = value['positive examples']
                #random.shuffle(pos)
                try:
                    datapoint_pos = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], pos))).to(device)) for e in embeddings], 1)
                except KeyError:
                    continue
                datapoints.append(datapoint_pos.unsqueeze(0))
                targets.append(key)
            targets_numerical = torch.zeros((len(targets), len(self.vocab), self.max_num_atom_repeat))
            for j in range(len(targets)):
                targets_numerical[j] = self.get_scores_of_atom_indices(targets[j])
            yield torch.cat(datapoints), targets_numerical, targets
        else:
            datapoints = []
            targets = []
            for key, value in data:
                pos = value['positive examples']
                #random.shuffle(pos)
                try:
                    datapoint_pos = torch.cat([e(torch.tensor(list(map(lambda x: self.entity_to_idx[x], pos))).to(device)) for e in embeddings], 1)
                except KeyError:
                    continue
                datapoints.append(datapoint_pos.unsqueeze(0))
                targets.append(key)
            targets_numerical = torch.zeros((len(targets), len(self.vocab), self.max_num_atom_repeat))
            for j in range(len(targets)):
                targets_numerical[j] = self.get_scores_of_atom_indices(targets[j])
            yield torch.cat(datapoints), targets_numerical, targets
            
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
            
