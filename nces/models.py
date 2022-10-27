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
        individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'integer', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        
        self.input_size = kwargs.input_size
        self.max_len = kwargs.max_length
        self.proj_dim = kwargs.proj_dim
        self.embedding_dim = kwargs.embedding_dim
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        
        self.lstm = nn.LSTM(self.input_size, self.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(self.proj_dim)
        self.fc1 = nn.Linear(2*self.proj_dim, self.proj_dim)
        self.fc2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc3 = nn.Linear(self.proj_dim, len(self.vocab)*self.max_len)
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Added values: ", values)
        return list(values)
        
        
    def forward(self, x1, x2):
        seq1, _ = self.lstm(x1)
        seq2, _ = self.lstm(x2)
        out1 = seq1.sum(1).view(-1, self.proj_dim)
        out2 = seq2.sum(1).view(-1, self.proj_dim)
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
        individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'integer', 'xsd']
        # 'string', 'boolean', 'float', 'decimal', 'dateTime', 'anyURI'
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        
        self.input_size = kwargs.input_size
        self.max_len = kwargs.max_length
        self.proj_dim = kwargs.proj_dim
        self.embedding_dim = kwargs.embedding_dim
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        
        self.gru = nn.GRU(self.input_size, self.proj_dim, kwargs.rnn_n_layers, dropout=kwargs.drop_prob, batch_first=True)
        self.bn = nn.BatchNorm1d(self.proj_dim)
        self.fc1 = nn.Linear(2*self.proj_dim, self.proj_dim)
        self.fc2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc3 = nn.Linear(self.proj_dim, len(self.vocab)*self.max_len)
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Added values: ", values)
        return list(values)
    
    def forward(self, x1, x2):
        seq1, _ = self.gru(x1)
        seq2, _ = self.gru(x2)
        out1 = seq1.sum(1).view(-1, self.proj_dim)
        out2 = seq2.sum(1).view(-1, self.proj_dim)
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
        individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                    'double', 'integer', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        
        self.input_size = kwargs.input_size
        self.max_len = kwargs.max_length
        self.proj_dim = kwargs.proj_dim
        self.embedding_dim = kwargs.embedding_dim
        self.ln = kwargs.ln
        self.num_heads = kwargs.num_heads
        self.num_inds = kwargs.num_inds
        self.num_seeds = kwargs.num_seeds
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        
        self.enc = nn.Sequential(
                ISAB(self.input_size, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln),
                ISAB(self.proj_dim, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln))
        self.dec = nn.Sequential(
                PMA(self.proj_dim, self.num_heads, self.num_seeds, ln=self.ln),
                nn.Linear(self.proj_dim, len(self.vocab)*self.max_len))
        
    def get_data_property_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        for ce in test_data:
            if '[' in ce:
                for val in re.findall(r"\[*-?\d*\.\d+]|\[*-?\d*]", ce):
                    values.add(val.strip(']'))
        print("*** Done! ***\n")
        print("Added values: ", values)
        return list(values)

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x
