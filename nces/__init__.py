import os, torch, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('base')[0])
import matplotlib.pyplot as plt
from ontolearn.knowledge_base import KnowledgeBase
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer
import json
import re

class BaseConceptSynthesis:
    """Supervised Machine Learning approach for learning class expressions in ALC from examples"""
    
    def __init__(self, kwargs):
        self.knowledge_base_path = kwargs.knowledge_base_path
        kb = KnowledgeBase(path=self.knowledge_base_path)
        self.__num_examples__ = min(kwargs.num_examples, kb.individuals_count()//2)
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        atomic_concepts = list(kb.ontology().classes_in_signature())
        atomic_concept_names = [self.dl_syntax_renderer.render(a) for a in atomic_concepts]
        role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                     [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
        vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                    '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']', 'double', 'xsd']
        quantified_restriction_values = [str(i) for i in range(1,12)]
        data_values = self.get_data_property_values(kwargs.knowledge_base_path)
        vocab = vocab + data_values + quantified_restriction_values
        vocab = sorted(vocab) + ['PAD']
        self.inv_vocab = vocab
        self.vocab = {vocab[i]:i for i in range(len(vocab))} #dict(map(reversed, enumerate(vocab)))
        self.max_length = kwargs.max_length
        self.kwargs = kwargs
        
    def get_data_property_values(self, path):
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
        return list(values)
        
    @property
    def num_examples(self):
        return self.__num_examples__
    
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