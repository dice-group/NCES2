import random
from tqdm import tqdm
class ConceptDescriptionGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, depth=2, max_length=10, num_rand_samples=150):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.depth = depth
        self.num_rand_samples = num_rand_samples
        self.max_length = max_length

    def apply_rho(self, concept):
        refinements = {ref for ref in self.rho.refine(concept, max_length=self.max_length)}
        if refinements:
            return list(refinements)

    def generate(self):
        roots = self.apply_rho(self.kb.thing)
        print ("|Thing refinements|: ", len(roots))
        Refinements = set(roots)
        for root in tqdm(random.sample(roots, k=self.num_rand_samples), desc="Sampling constructs..."):
            current_state = root
            for _ in range(self.depth):
                #try:
                refts = self.apply_rho(current_state)
                current_state = random.sample(refts, 1)[0] if refts else None
                if current_state is None:
                    break
                Refinements.update(refts)
#                 except AttributeError:
#                     pass
        return Refinements
