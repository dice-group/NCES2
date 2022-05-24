import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('train_data')[0])
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('generators')[0])

from data_generator import KBToDataForConceptSynthesis
from helper_classes.embedding_triples import RDFTriples

kb_path = os.path.dirname(os.path.realpath(__file__)).split('generators')[0]+"Datasets/semantic_bible/semantic_bible.owl"
kb_path1 = os.path.dirname(os.path.realpath(__file__)).split('generators')[0]+"Datasets/semantic_bible/NTNcombined.owl"

triples = RDFTriples(source_kg_path=kb_path1)
triples.export_triples()


kb_to_data = KBToDataForConceptSynthesis(path=kb_path, concept_gen_path_length=4, max_child_length=8, refinement_expressivity=0.7, downsample_refinements=True, num_rand_samples=300, min_num_pos_examples=1, max_num_pos_examples=500, num_examples = 362)

kb_to_data.generate_descriptions().save_train_data()