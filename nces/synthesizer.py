import torch
import torch.nn as nn
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('nces')[0]
sys.path.append(base_path)
from .models import *
from embeddings.util.complex_models import *
from embeddings.util.real_models import *
import pandas as pd

class ConceptSynthesizer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.learner_name = kwargs.learner_name
        self.kb_emb_model = kwargs.kb_emb_model
        self.model = self.get_synthesizer()
        self.embedding_model = self.get_embedding_model()
    
    def get_synthesizer(self):
        if self.learner_name == 'SetTransformer':
            return SetTransformer(self.kwargs)
        elif self.learner_name == 'GRU':
            return ConceptLearner_GRU(self.kwargs)
        elif self.learner_name == 'LSTM':
            return ConceptLearner_LSTM(self.kwargs)
        elif self.learner_name == 'TreeTransformer':
            return TreeTransformer(self.kwargs)
        else:
            print('Wrong concept learner name')
            raise ValueError
            
            
    def get_embedding_model(self):
        if self.kb_emb_model == 'ConEx':
            return ConEx(self.kwargs)
        elif self.kb_emb_model == 'Complex':
            return Complex(self.kwargs)
        elif self.kb_emb_model == 'Distmult':
            return Distmult(self.kwargs)
        elif self.kb_emb_model == 'Tucker':
            return Tucker(self.kwargs)
        else:
            print('No embedding model given, will require pretrained embeddings in csv format')
            
            
    def refresh(self):
        self.model = self.get_synthesizer()
        
    def load_pretrained(self):
        assert self.kwargs.pretrained_concept_synthesizer, 'No pretrained length learner'
        self.model = torch.load(self.kwargs.pretrained_concept_synthesizer, map_location=torch.device('cpu'))
        self.model.eval()
           
    