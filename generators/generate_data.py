import sys, os
from concept_description import ConceptDescriptionGenerator
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement
import random, os, copy, json
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer
from sklearn.utils import resample
from tqdm import tqdm

from sklearn.model_selection import train_test_split

random.seed(42)

class KBToDataForConceptSynthesis:
    """
    This class takes an owl file, loads it into a knowledge base using ontolearn.knowledge_base.KnowledgeBase.
    A refinement operator is used to generate a large number of class expressions, which we filter and retain the shortest non-redundant ones.
   Finally, we aim at training a deep neural network to predict the syntax of class expressions from their instances/examples. 
   Hence, we export each concept and its examples (eventually positive and negative examples) into json files.  
    """

    def __init__(self, path, rho_name="ExpressRefinement", depth=5, max_child_length=25, refinement_expressivity=0.6, downsample_refinements=True, k=5, num_rand_samples=150, min_num_pos_examples=1, max_num_pos_examples=2000):
        self.path = path
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.kb = KnowledgeBase(path=path)
        self.num_examples = min(self.kb.individuals_count()//2, 1000)
        self.min_num_pos_examples = min_num_pos_examples
        self.max_num_pos_examples = max_num_pos_examples
        atomic_concepts: Final = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, k=k, downsample = downsample_refinements, expressivity=refinement_expressivity) if \
        rho_name == "ExpressRefinement" else ModifiedCELOERefinement(knowledge_base=self.kb)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, depth=depth, num_rand_samples=num_rand_samples)

    
    def generate_descriptions(self):
        print()
        print("#"*60)
        print("Started generating data on the "+self.path.split("/")[-1].split(".")[0]+" knowledge base")
        print("#"*60)
        print()
        All_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge base: {} \n".format(len(All_individuals)))
        Concepts = self.lp_gen.generate()
        non_redundancy_hash_map = dict()
        show_some_length = True
        for concept in sorted(Concepts, key=lambda c: self.kb.concept_len(c)):
            if not self.kb.individuals_set(concept) in non_redundancy_hash_map and self.min_num_pos_examples <= self.kb.individuals_count(concept) <= self.max_num_pos_examples:
                non_redundancy_hash_map[self.kb.individuals_set(concept)] = concept
            else: continue
            if self.kb.concept_len(concept) >= 15 and show_some_length:
                print("A long concept found: ", self.kb.concept_len(concept))
                show_some_length = False
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ", max({l for l in [self.kb.concept_len(c) for c in non_redundancy_hash_map.values()]}), "\n")
        print("Total number of concepts: ", len(non_redundancy_hash_map), "\n")
        self.train_concepts = list(non_redundancy_hash_map.values())
        print("Data generation completed")
        return self
    

    def save_data(self):
        data = dict()
        for concept in tqdm(self.train_concepts):
            pos = set(self.kb.individuals(concept))
            neg = set(self.kb.individuals())-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            if min(len(neg),len(pos)) >= self.num_examples//2:
                if len(pos) > len(neg):
                    num_neg_ex = self.num_examples//2
                    num_pos_ex = self.num_examples-num_neg_ex
                else:
                    num_pos_ex = self.num_examples//2
                    num_neg_ex = self.num_examples-num_pos_ex
            elif len(pos) > len(neg):
                num_neg_ex = len(neg)
                num_pos_ex = self.num_examples-num_neg_ex
            elif len(pos) < len(neg):
                num_pos_ex = len(pos)
                num_neg_ex = self.num_examples-num_pos_ex
            else:
                print("Invalid number of instances")
                continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            data[concept_name] = {'positive examples': positive, 'negative examples': negative}
            
        data = list(data.items())
        data_train, data_test = train_test_split(data, test_size=0.01, random_state=42)
        os.makedirs(f'../datasets/{self.path.split("/")[-2]}/Train_data/', exist_ok=True)
        os.makedirs(f'../datasets/{self.path.split("/")[-2]}/Test_data/', exist_ok=True)
        with open(f'../datasets/{self.path.split("/")[-2]}/Test_data/Data.json', 'w') as file_test:
            json.dump(dict(data_test), file_test, indent=3, ensure_ascii=False)

        with open(f'../datasets/{self.path.split("/")[-2]}/Train_data/Data.json', 'w') as file_train:
            json.dump(dict(data_train), file_train, indent=3, ensure_ascii=False)

        print(f'Data saved at ../datasets/{self.path.split("/")[-2]}')
              
            
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name')
parser.add_argument('--num_rand_samples', type=int, default=100, help='The number of random samples at each step of the generation process')
parser.add_argument('--depth', type=int, default=6, help='The depth of refinements')
parser.add_argument('--k', type=int, default=5, help='The number of fillers to sample')
parser.add_argument('--max_child_len', type=int, default=15, help='Maximum child length')
parser.add_argument('--max_num_pos_examples', type=int, default=100000, help='Maximum number of positive examples')
parser.add_argument('--refinement_expressivity', type=float, default=0.5)
parser.add_argument('--rho', type=str, default='ExpressRefinement', choices=['ExpressRefinement', 'ModifiedCELOERefinement'], help='Refinement operator to use')

args = parser.parse_args()

for kb in args.kbs:
    with open(f'../datasets/{kb}/data_generation_settings.json', "w") as setting:
        json.dump(vars(args), setting)
    DataGen = KBToDataForConceptSynthesis(path=f'../datasets/{kb}/{kb}.owl', rho_name=args.rho, depth=args.depth, k=args.k, max_child_length=args.max_child_len, refinement_expressivity=args.refinement_expressivity, downsample_refinements=True, num_rand_samples=args.num_rand_samples, min_num_pos_examples=1, max_num_pos_examples=args.max_num_pos_examples)
    DataGen.generate_descriptions().save_data()