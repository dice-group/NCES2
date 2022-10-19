import random, json
from collections import defaultdict, Counter
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
import os

random.seed(42)

class LearningProblemGenerator:
    """
    Learning problem generator.
    """
    
    def __init__(self, kb_path=None, rho='ExpressRefinement', depth=2, num_rand_samples=150, max_child_len=6, refinement_expressivity=0.1, min_num_pos_examples=1, max_num_pos_examples=1000):
        assert kb_path is not None, "Provide a path for the knowledge base"
        self.kb = KnowledgeBase(path=kb_path)
        if rho == 'ExpressRefinement':
            self.rho = ExpressRefinement(self.kb, max_child_length=max_child_len, expressivity=refinement_expressivity)
        else:
            self.rho = ModifiedCELOERefinement(self.kb)
        self.rho_name = rho
        self.depth = depth
        self.num_rand_samples = num_rand_samples
        self.min_num_pos_examples = min_num_pos_examples
        self.max_num_pos_examples = max_num_pos_examples
        self.max_len = max_child_len
        self.dl_syntax_renderer = DLSyntaxObjectRenderer()
        self.path = kb_path
        
        atomic_concepts = frozenset(self.kb.ontology().classes_in_signature())
        self.atomic_concept_names: Final = frozenset([self.dl_syntax_renderer.render(a) for a in atomic_concepts])
        
                
    def apply_rho(self, concept):
        refinements = {ref for ref in self.rho.refine(concept, max_length=self.max_len,
                                                      current_domain=concept)}
        if refinements:
            return list(refinements)
        
    def generate(self):
        print()
        print("#"*70)
        print("Started generating learning problems on %s" % self.path.split("/")[-1].split(".")[0]+" knowledge base")
        print("#"*70)
        roots = self.apply_rho(self.kb.thing)
        print ("|Thing refinements|: ", len(roots))
        Refinements = set()
        Refinements.update(roots)
        if self.num_rand_samples == 0:
            return Refinements
        for root in random.sample(roots, k=self.num_rand_samples):
            current_state = root
            for _ in range(self.depth):
                refts = self.apply_rho(current_state)
                current_state = random.choice(refts) if refts else None
                if current_state is None:
                    break
                Refinements.update(refts)
        return Refinements


    
    def Filter(self):
        self.learning_problems = defaultdict(lambda : defaultdict(list))
        All_individuals = set(self.kb.individuals())
        print("Number of individuals in the knowledge base: {} \n".format(len(All_individuals)))
        generated_concept_descriptions = sorted(self.generate(), key=lambda c: self.kb.concept_len(c))
        cardinality = len(generated_concept_descriptions)
        print('\n Number of concept descriptions generated: ', cardinality, "\n")
        count = 0
        for concept in generated_concept_descriptions:
            pos = set(self.kb.individuals(concept))
            neg = All_individuals-pos
            pos = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
            neg = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
            count += 1
            if not self.min_num_pos_examples <= len(pos) <= self.max_num_pos_examples:
                continue
            if count % 100 == 0:
                print('Progress: ', 100 * float(count)/cardinality, "%")
            self.learning_problems[self.dl_syntax_renderer.render(concept)]['positive examples'].extend(pos)
            self.learning_problems[self.dl_syntax_renderer.render(concept)]['negative examples'].extend(neg)
        return self
            
    def save_learning_problems(self):
        data = list(self.learning_problems.items())
        data = random.sample(data, min(len(data), 100))
        data = dict(data)
        if not os.path.exists(self.path[:self.path.rfind("/")]+"/Learning_problems"):
            os.mkdir(self.path[:self.path.rfind("/")]+"/Learning_problems")
        type_lp = 'learning_problems_celoe.json' if self.rho_name == 'ModifiedCELOERefinement' else 'learning_problems.json'
        with open(self.path[:self.path.rfind("/")]+"/Learning_problems/"+type_lp, "w") as file:
            json.dump(data, file, ensure_ascii=False, indent=3)
        print("{} learning problems saved at {}".format(len(data), self.path[:self.path.rfind("/")]+"/Learning_problems/"))
            
            
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
parser.add_argument('--num_rand_samples', type=int, default=10, help='The number of random samples at each step of the generation process')
parser.add_argument('--depth', type=int, default=4, help='The depth of refinements')
parser.add_argument('--max_child_len', type=int, default=6, help='Maximum child length')
parser.add_argument('--refinement_expressivity', type=float, default=0.5)
parser.add_argument('--rho', type=str, default='ExpressRefinement', choices=['ExpressRefinement', 'CELOERefinement'], help='Refinement operator to use')

args = parser.parse_args()

for kb in args.kbs:
    LPGen = LearningProblemGenerator(kb_path=f'../datasets/{kb}/{kb}.owl', rho=args.rho, depth=args.depth, num_rand_samples=args.num_rand_samples,\
                                    max_child_len=args.max_child_len, refinement_expressivity=args.refinement_expressivity)
    LPGen.Filter().save_learning_problems()