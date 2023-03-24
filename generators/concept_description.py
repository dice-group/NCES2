import random
from tqdm import tqdm

class ConceptDescriptionGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, max_length=10, num_rand_samples=150):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_rand_samples = num_rand_samples
        self.max_length = max_length

    def apply_rho(self, concept):
        return {ref for ref in self.rho.refine(concept, max_length=self.max_length)}

    def generate(self):
        roots = self.apply_rho(self.kb.thing)
        ## Now remove redundant expressions before proceeding!
        non_redundancy_hash_map = dict()
        show_some_length = True
        for concept in tqdm(roots, desc="Removing redundancy in refinements of Thing..."):
            length = self.kb.concept_len(concept)
            ind_set = self.kb.individuals_set(concept)
            if ind_set in non_redundancy_hash_map:
                if length < self.kb.concept_len(non_redundancy_hash_map[ind_set]):
                    non_redundancy_hash_map[ind_set] = concept
            else:
                non_redundancy_hash_map[ind_set] = concept
        roots = set(non_redundancy_hash_map.values())
        Refinements = set()
        Refinements.update(roots)
        print ("|Thing refinements|: ", len(roots))
        roots_sample = random.sample(list(roots), k=self.num_rand_samples)
        print("Number of roots: ", len(roots_sample))
        for root in tqdm(roots_sample, desc="Refining roots..."):
            Refinements.update(self.apply_rho(root))
        return Refinements
