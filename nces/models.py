import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from .modules import *
import sys, os, json
#base_path = os.path.dirname(os.path.realpath(__file__)).split('concept_synthesis')[0]
#sys.path.append(base_path)
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import pandas as pd
import re

class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(kwargs.input_size, kwargs.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs.proj_dim)
        self.fc1 = nn.Linear(2*kwargs.proj_dim, kwargs.proj_dim)
        self.fc2 = nn.Linear(kwargs.proj_dim, kwargs.proj_dim)
        self.fc3 = nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length)
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Added values: ", values)
        return list(values)
        
        
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.lstm(x1)
        seq2, _ = self.lstm(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs.proj_dim)
        out2 = seq2.sum(1).view(-1, self.kwargs.proj_dim)
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x

        
class ConceptLearner_GRU(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU'
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.gru = nn.GRU(kwargs.input_size, kwargs.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs.proj_dim)
        self.fc1 = nn.Linear(2*kwargs.proj_dim, kwargs.proj_dim)
        self.fc2 = nn.Linear(kwargs.proj_dim, kwargs.proj_dim)
        self.fc3 = nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length)
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Added values: ", values)
        return list(values)
    
    def forward(self, x1, x2, target_scores=None):
        seq1, _ = self.gru(x1)
        seq2, _ = self.gru(x2)
        out1 = seq1.sum(1).view(-1, self.kwargs.proj_dim)
        out2 = seq2.sum(1).view(-1, self.kwargs.proj_dim)
        x = torch.cat([out1,out2], 1)
        x = F.gelu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)
        x = x.reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x

    
    
class SetTransformer(nn.Module):
    def __init__(self, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.kwargs = kwargs
        kb = KnowledgeBase(path=kwargs.knowledge_base_path)
        renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        self.max_len = kwargs.max_length
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        self.enc = nn.Sequential(
                ISAB(kwargs.input_size, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln),
                ISAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln))
        self.dec = nn.Sequential(
                PMA(kwargs.proj_dim, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                nn.Linear(kwargs.proj_dim, len(self.vocab)*kwargs.max_length))
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d\.\d+]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Values found: ", values)
        return list(values)

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x
