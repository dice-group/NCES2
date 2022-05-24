import os, torch, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('base')[0])
import matplotlib.pyplot as plt
from ontolearn.knowledge_base import KnowledgeBase
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer

class BaseConceptSynthesis:
    """Supervised Machine Learning approach for learning class expressions in ALC from examples"""
    
    def __init__(self, kwargs):
        self.knowledge_graph_path = kwargs['knowledge_graph_path']
        kb = KnowledgeBase(path=self.knowledge_graph_path)
        self.__num_examples__ = min(kwargs['num_examples'], kb.individuals_count())
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.index_score_upper_bound = kwargs['index_score_upper_bound']
        self.index_score_lower_bound_rate = kwargs['index_score_lower_bound_rate']
        atomic_concepts: Final = frozenset(kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        self.role_names: Final = frozenset([rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()])
        vocab = list(self.atomic_concept_names) + list(self.role_names) + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ']
        self.vocab = {vocab[i]:i for i in range(len(vocab))}#dict(map(reversed, enumerate(vocab)))
        self.max_num_atom_repeat: Final[int] = kwargs['max_num_atom_repeat']
        self.kwargs = kwargs
        
    @property
    def num_examples(self):
        return self.__num_examples__
    
    @staticmethod
    def decompose(concept_name: str) -> list:
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in ['(', ')', '⊔', '⊓', '∃', '∀', '¬', '.', ' ']:
                concept += concept_name[i]
                i += 1
            if concept and i < len(concept_name) and not concept_name[i] in ['(', ')']:
                list_ordered_pieces.extend([concept, concept_name[i]])
            elif concept:
                list_ordered_pieces.append(concept)
            elif i < len(concept_name) and not concept_name[i] in ['(', ')']:
                list_ordered_pieces.append(concept_name[i])
            i += 1
        return list_ordered_pieces
    
    def get_scores_of_atom_indices(self, target):
        Scores = torch.zeros((len(self.vocab),self.max_num_atom_repeat))
        target = self.decompose(target)
        scores = torch.tensor(len(target)).sqrt()*torch.linspace(self.index_score_upper_bound, self.index_score_upper_bound*(1-self.index_score_lower_bound_rate), len(target))
        atom_counts = {a: 0 for a in target}
        for j in range(len(target)):
            try:
                Scores[self.vocab[target[j]], atom_counts[target[j]]] = scores[j]
                atom_counts[target[j]] += 1
            except IndexError:
                print('Index out of bound error, ignoring current atom index')
        return Scores