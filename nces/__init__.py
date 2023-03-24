import os, torch, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('base')[0])
from ontolearn.knowledge_base import KnowledgeBase
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer
import json
import re
import copy

class BaseConceptSynthesis:
    """Supervised Machine Learning approach for learning class expressions in ALC from examples"""
    
    def __init__(self, vocab, inv_vocab, kwargs):
        self.max_length = kwargs.max_length
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.kwargs = kwargs
        
    @staticmethod
    def decompose(concept_name: str) -> list:
        """ Decomposes a class expression into a sequence of tokens (atoms) """
        def is_number(char):
            """ Checks if a character can be converted into a number """
            try:
                int(char)
                return True
            except:
                return False
        specials = ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ' ', '(', ')',\
                    '⁻', '≤', '≥', '{', '}', ':', '[', ']']
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in specials:
                if concept_name[i] == '.' and not is_number(concept_name[i-1]):
                    break
                concept += concept_name[i]
                i += 1
            if concept and i < len(concept_name):
                list_ordered_pieces.extend([concept, concept_name[i]])
            elif concept:
                list_ordered_pieces.append(concept)
            elif i < len(concept_name):
                list_ordered_pieces.append(concept_name[i])
            i += 1
        return list_ordered_pieces
    
    
    def get_labels(self, target):
        target = self.decompose(target)
        labels = [self.vocab[atm] for atm in target]
        return labels, len(target)