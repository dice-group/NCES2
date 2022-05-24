
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('learning_problems')[0])

from learning_problem_generator import LearningProblemGenerator
from ontolearn import KnowledgeBase
from ontolearn.refinement_operators import ExpressRefinement

kb_path = os.path.dirname(os.path.realpath(__file__)).split('generators')[0]+"Datasets/semantic_bible/semantic_bible.owl"
kb = KnowledgeBase(path=kb_path)
lpg = LearningProblemGenerator(kb_path=kb_path, rho='ModifiedCELOERefinement', depth=5, num_rand_samples=50, max_ref_child_length=15, refinement_expressivity=0.2, min_num_pos_examples=1, max_num_pos_examples=500)
lpg.Filter().save_learning_problems()