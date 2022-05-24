import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('train_data')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])

from data_generator import KBToDataForConceptSynthesis
from helper_classes.embedding_triples import RDFTriples

kb_path = os.path.dirname(os.path.realpath(__file__)).split('generators')[0]+"Datasets/family-benchmark/family-benchmark_rich_background.owl"

triples = RDFTriples(source_kg_path=kb_path)
triples.export_triples()

kb_to_data = KBToDataForConceptSynthesis(path=kb_path, concept_gen_path_length=4, max_child_length=8, refinement_expressivity=0.9, downsample_refinements=True, num_rand_samples=800, min_num_pos_examples=1, max_num_pos_examples=150, num_examples = 101)

kb_to_data.generate_descriptions().save_train_data()