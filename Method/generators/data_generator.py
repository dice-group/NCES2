import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])

from generators.concept_description import ConceptDescriptionGenerator
from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement
import random, os, copy, json
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer
from sklearn.utils import resample

random.seed(1)
class KBToDataForConceptSynthesis:
    """
    This class takes an owl file, loads it into a knowledge base using ontolearn.knowledge_base.KnowledgeBase.
    A refinement operator is used to generate a large number of concepts, from which we filter and retain the shortest non-redundant concepts.
   Finally, we aim at training a deep neural network to predict the syntax of concepts from their instances. Hence, we export each concept and its instances (eventually positive and negative examples) into json files.  
    """

    def __init__(self, path, concept_gen_path_length=5, max_child_length=25, refinement_expressivity=0.6, downsample_refinements=True, num_rand_samples=150, min_num_pos_examples=1, max_num_pos_examples=2000, num_examples=1000):
        self.path = path
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.kb = KnowledgeBase(path=path)
        self.num_examples = num_examples
        self.min_num_pos_examples = min_num_pos_examples
        self.max_num_pos_examples = max_num_pos_examples
        atomic_concepts: Final = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        self.role_names: Final = frozenset([rel.get_iri().get_remainder() for rel in self.kb.ontology().object_properties_in_signature()]) #+list(kb.ontology().data_properties_in_signature()) data properties are not yet supported
        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, downsample = downsample_refinements, expressivity=refinement_expressivity)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, depth=concept_gen_path_length, num_rand_samples=num_rand_samples)

    
    def generate_descriptions(self):
        print()
        print("#"*60)
        print("Started generating data on the "+self.path.split("/")[-1].split(".")[0]+" knowledge graph")
        print("#"*60)
        print()
        All_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge graph: {} \n".format(len(All_individuals)))
        Concepts = self.lp_gen.generate()
        non_redundancy_hash_map = dict()
        show_some_length = True
        for concept in sorted(Concepts, key=lambda c: self.kb.cl(c)):
            if not self.kb.individuals_set(concept) in non_redundancy_hash_map and self.min_num_pos_examples <= self.kb.individuals_count(concept) <= self.max_num_pos_examples:
                non_redundancy_hash_map[self.kb.individuals_set(concept)] = concept
            else: continue
            if self.kb.cl(concept) >= 15 and show_some_length:
                print("A long concept found: ", self.kb.cl(concept))
                show_some_length = False
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ", max({l for l in [self.kb.cl(c) for c in non_redundancy_hash_map.values()]}), "\n")
        print("Total number of concepts: ", len(non_redundancy_hash_map), "\n")
        self.train_concepts = list(non_redundancy_hash_map.values())
        print("Data generation completed")
        return self
    

    def save_train_data(self):
        data = dict()
        for concept in self.train_concepts:
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in self.kb.individuals(concept)]
            if len(pos) < self.num_examples:
                pos = resample(pos, replace=True, n_samples=self.num_examples, random_state=123)
            else:
                pos = random.sample(pos, k=self.num_examples)
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            concept_length = self.kb.cl(concept)
            data[concept_name] = {'positive examples': pos, 'concept length': concept_length}
        if not os.path.exists('/'+("/").join(self.path.split("/")[1:-1])+"/"+"Train_data/"):
            os.mkdir('/'+("/").join(self.path.split("/")[1:-1])+"/"+"Train_data/")
        with open('/'+("/").join(self.path.split("/")[1:-1])+"/"+"Train_data/Data.json", 'w') as file_descriptor:
            json.dump(data, file_descriptor, ensure_ascii=False, indent=3)
        print("Data saved at %s"% "/"+("/").join(self.path.split("/")[1:-1]))
              
            
