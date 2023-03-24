import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from .modules import *
import json
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import pandas as pd
import re

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
                                                    'double', 'integer', 'date', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.add_data_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(set(vocab)) + ['PAD']
        print("Vocabulary size: ", len(vocab))
        self.num_examples = min(kwargs.num_examples, kb.individuals_count()//2)
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
                ISAB(self.embedding_dim, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln),
                ISAB(self.proj_dim, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln))
        self.dec = nn.Sequential(
                PMA(self.proj_dim, self.num_heads, self.num_seeds, ln=self.ln),
                nn.Linear(self.proj_dim, len(self.vocab)*self.max_len))
        
    def add_data_values(self, path):
        print("\n*** Finding relevant data values ***")
        with open(path[:path.rfind("/")+1]+"Train_data/Data.json") as file_train:
            train_data = json.load(file_train)
        with open(path[:path.rfind("/")+1]+"Test_data/Data.json") as file_test:
            test_data = json.load(file_test)
        values = set()
        for ce in train_data+test_data:
            ce = ce[0]
            if '[' in ce:
                for val in re.findall("\[(.*?)\]", ce):
                    values.add(val.split(' ')[-1])
        print("*** Done! ***\n")
        print("Added values: ", values)
        print()
        return list(values)

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x
