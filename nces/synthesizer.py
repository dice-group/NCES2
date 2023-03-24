import torch
import torch.nn as nn
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('nces')[0]
sys.path.append(base_path)
from .models import SetTransformer
from embeddings.util.complex_models import *
from embeddings.util.real_models import *
import pandas as pd

class ConceptSynthesizer:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.learner_name = kwargs.learner_name
        self.kb_emb_model = kwargs.kb_emb_model
    
    def get_synthesizer(self):
        try:
            assert self.learner_name == "SetTransformer", "Wrong model name, ignoring..."
        except:
            pass
        return SetTransformer(self.kwargs)
            
            
    def get_embedding_model(self):
        if self.kb_emb_model.lower() == 'conex':
            return ConEx(self.kwargs)
        elif self.kb_emb_model.lower() == 'complex':
            return Complex(self.kwargs)
        elif self.kb_emb_model.lower() == 'distmult':
            return Distmult(self.kwargs)
        elif self.kb_emb_model.lower() == 'tucker':
            return Tucker(self.kwargs)
        else:
            print('No embedding model given, will require pretrained embeddings in csv format')
            
            
    def refresh(self):
        self.model = self.get_synthesizer()
        self.embedding_model = self.get_embedding_model()
        
    def load_pretrained(self, path_synthesizer, path_embedding):
        self.model = torch.load(path_synthesizer, map_location=torch.device('cpu'))
        self.embedding_model = torch.load(path_embedding, map_location=torch.device('cpu'))
           
    