import torch
import torch.nn as nn
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('nces')[0]
sys.path.append(base_path)
from .models import *
from owlapy.model import OWLNamedIndividual
from typing import List, Union
import pandas as pd

class ConceptSynthesizer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.learner_name = kwargs.learner_name
        self.model = self.get_synthesizer()
        self.embeddings = self.get_embedding()
        
    
    def get_embedding(self):
        return pd.read_csv(self.kwargs.path_to_csv_embeddings).set_index('Unnamed: 0')
    
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
            
    def refresh(self):
        self.model = self.get_synthesizer()
        
    def load_pretrained(self):
        assert self.kwargs.pretrained_concept_synthesizer, 'No pretrained length learner'
        self.model = torch.load(self.kwargs.pretrained_concept_synthesizer, map_location=torch.device('cpu'))
        self.model.eval()
           
    