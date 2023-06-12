import sys, os, numpy as np
from concept_description import ConceptDescriptionGenerator
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement
import random, os, copy, json
from typing import Final
from owlapy.render import DLSyntaxObjectRenderer
from sklearn.utils import resample
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
random.seed(42)

class KBToDataForConceptSynthesis:
    """
    This class takes an owl file, loads it into a knowledge base using ontolearn.knowledge_base.KnowledgeBase.
    A refinement operator is used to generate a large number of class expressions, which we filter and retain the shortest non-redundant ones.
   Finally, we aim at training a deep neural network to predict the syntax of class expressions from their instances/examples. 
   Hence, we export each concept and its examples (eventually positive and negative examples) into json files.  
    """

    def __init__(self, path, rho_name="ExpressRefinement", max_child_length=25, refinement_expressivity=0.6, downsample_refinements=True, k=5, num_rand_samples=150):
        self.path = path
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.kb = KnowledgeBase(path=path)
        self.total_num_inds = self.kb.individuals_count()
        self.num_examples = min(self.total_num_inds//2, 1000)
        atomic_concepts: Final = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        rho = ExpressRefinement(knowledge_base=self.kb, max_child_length=max_child_length, sample_fillers_count=k, downsample=downsample_refinements, expressivity=refinement_expressivity) if \
        rho_name == "ExpressRefinement" else ModifiedCELOERefinement(knowledge_base=self.kb)
        self.lp_gen = ConceptDescriptionGenerator(knowledge_base=self.kb, refinement_operator=rho, num_rand_samples=num_rand_samples)
        
    def generate_descriptions(self):
        print()
        print("#"*60)
        print("Started generating data on the "+self.path.split("/")[-1].split(".")[0]+" knowledge base")
        print("#"*60)
        print()
        all_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge base: {} \n".format(len(all_individuals)))
        concepts = list(self.lp_gen.generate())
        non_redundancy_hash_map = dict()
        show_some_length = True
        for concept in tqdm(concepts, desc="Removing redundancy..."):
            length = self.kb.concept_len(concept)
            ind_set = self.kb.individuals_set(concept)
            if ind_set in non_redundancy_hash_map:
                if length < self.kb.concept_len(non_redundancy_hash_map[ind_set]):
                    non_redundancy_hash_map[ind_set] = concept
            else:
                non_redundancy_hash_map[ind_set] = concept
            
        print("Concepts generation done!\n")
        print("Number of atomic concepts: ", len(self.atomic_concept_names))
        print("Longest concept length: ", max({l for l in [self.kb.concept_len(concept) for concept in non_redundancy_hash_map.values()]}), "\n")
        print("Total number of concepts: ", len(non_redundancy_hash_map), "\n")
        data_train, data_test = train_test_split(list(non_redundancy_hash_map.values()), test_size=0.01, random_state=42)
        print("Data generation completed")
        return data_train, data_test
    
    def find_sampling_sizes(self, pos, neg):
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
        return num_pos_ex, num_neg_ex
    
    def reinforced_example_sampling(self, data_train, data_test, num_subsamples=2):
        """ Robust sampling to make sure that we cover many examples for each learning problem: some learning problems contain thousands (or millions on large datasets) of examples, and we aim to sample examples. As a result, we sample several times instead of just one time as in naive_example_sampling.
        """
        def sample(pos, neg, n_pos, n_neg):
            Pos = []
            Neg = []
            for _ in range(num_subsamples):
                Pos.append(random.sample(pos, n_pos))
                Neg.append(random.sample(neg, n_neg))
            return Pos, Neg
        final_data_train = []
        final_data_test = []
        all_individuals = set(self.kb.individuals())
        for concept in tqdm(data_train, desc="Processing and writing training data..."):
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            concept_length = self.kb.concept_len(concept)
            pos = set(self.kb.individuals(concept))
            neg = all_individuals-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            num_pos_ex, num_neg_ex = self.find_sampling_sizes(pos, neg)
            if num_pos_ex * num_neg_ex == 0:
                continue # Extreme cases where there are no positive exaples or negative examples
            Pos, Neg = sample(pos, neg, num_pos_ex, num_neg_ex)
            for p, n in zip(Pos, Neg):
                final_data_train.append([concept_name, {'positive examples': p, 'negative examples': n, 'length': concept_length}])
        for concept in tqdm(data_test, desc="Processing and writing test data..."):
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            concept_length = self.kb.concept_len(concept)
            pos = set(self.kb.individuals(concept))
            neg = all_individuals-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            num_pos_ex, num_neg_ex = self.find_sampling_sizes(pos, neg)
            if num_pos_ex * num_neg_ex == 0: continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            final_data_test.append([concept_name, {'positive examples': positive, 'negative examples': negative, 'length': concept_length}])
        return final_data_train, final_data_test
        
    def naive_example_sampling(self, data_train, data_test):
        final_data_train = []
        final_data_test = []
        all_individuals = set(self.kb.individuals())
        for concept in tqdm(data_train, desc="Processing and writing training data..."):
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            concept_length = self.kb.concept_len(concept)
            pos = set(self.kb.individuals(concept))
            neg = all_individuals-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            num_pos_ex, num_neg_ex = self.find_sampling_sizes(pos, neg)
            if num_pos_ex * num_neg_ex == 0: continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            final_data_train.append([concept_name, {'positive examples': positive, 'negative examples': negative, 'length': concept_length}])
        for concept in tqdm(data_test, desc="Processing and writing test data..."):
            concept_name = self.dl_syntax_renderer.render(concept.get_nnf())
            concept_length = self.kb.concept_len(concept)
            pos = set(self.kb.individuals(concept))
            neg = all_individuals-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            num_pos_ex, num_neg_ex = self.find_sampling_sizes(pos, neg)
            if num_pos_ex * num_neg_ex == 0: continue
            positive = random.sample(pos, num_pos_ex)
            negative = random.sample(neg, num_neg_ex)
            final_data_test.append([concept_name, {'positive examples': positive, 'negative examples': negative, 'length': concept_length}])
        return final_data_train, final_data_test
        
            
    def save_data(self, data_train, data_test):
        os.makedirs(f'../datasets/{self.path.split("/")[-2]}/Train_data/', exist_ok=True)
        os.makedirs(f'../datasets/{self.path.split("/")[-2]}/Test_data/', exist_ok=True)
        with open(f'../datasets/{self.path.split("/")[-2]}/Test_data/Data.json', 'w') as file_test:
            json.dump(data_test, file_test, indent=3, ensure_ascii=False)

        with open(f'../datasets/{self.path.split("/")[-2]}/Train_data/Data.json', 'w') as file_train:
            json.dump(data_train, file_train, indent=3, ensure_ascii=False)

        print(f'Data saved at ../datasets/{self.path.split("/")[-2]}')
              

if __name__ == '__main__':            
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name')
    parser.add_argument('--num_rand_samples', type=int, default=300, help='The number of random samples at each step of the generation process')
    parser.add_argument('--k', type=int, default=10, help='The number of fillers to sample')
    parser.add_argument('--max_child_len', type=int, default=15, help='Maximum child length')
    parser.add_argument('--refinement_expressivity', type=float, default=0.6)
    parser.add_argument('--rho', type=str, default='ExpressRefinement', choices=['ExpressRefinement', 'ModifiedCELOERefinement'], help='Refinement operator to use')

    args = parser.parse_args()

    for kb in args.kbs:
        with open(f'../datasets/{kb}/data_generation_settings.json', "w") as setting:
            json.dump(vars(args), setting)
        DataGen = KBToDataForConceptSynthesis(path=f'../datasets/{kb}/{kb}.owl', rho_name=args.rho, k=args.k, max_child_length=args.max_child_len, refinement_expressivity=args.refinement_expressivity, downsample_refinements=True, num_rand_samples=args.num_rand_samples)
        data_train, data_test = DataGen.generate_descriptions()
        data_train, data_test = DataGen.reinforced_example_sampling(data_train, data_test)
        DataGen.save_data(data_train, data_test)