import json
import os, sys
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("evolearner")[0])
from utils.evaluator import Evaluator
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.model import OWLClass, OWLNamedIndividual, IRI
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from ontolearn.utils import setup_logging
import argparse
import urllib.parse
from tqdm import tqdm
from collections import defaultdict
import time
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Whether to print the target and predicted class expressions')
    args = parser.parse_args()

    dlsr = DLSyntaxObjectRenderer()
    
    for kb_name in args.kbs:
        print(f"\n### On {kb_name.upper()} ###\n")
        with open(f'../datasets/{kb_name}/Test_data/Data.json') as json_file:
            lps = json.load(json_file)
        kb = KnowledgeBase(path=f'../datasets/{kb_name}/{kb_name}.owl')
        kb_namespace = list(kb.individuals())[0].get_iri().get_namespace()
        kb_prefix = kb_namespace[:kb_namespace.rfind("/")+1]
        evaluator = Evaluator(kb)
        dl_parser = DLSyntaxParser(namespace = kb_namespace)
        all_individuals = set(kb.individuals())
        Result_dict = {'F-measure': [], 'Accuracy': [], 'Runtime': [], 'Prediction': [], 'Learned Concept': []}
        Avg_result = defaultdict(lambda: defaultdict(float))
        for str_target_concept, examples in tqdm(lps, desc=f'Learning {len(lps)} problems'):
            model = EvoLearner(knowledge_base=kb, max_runtime=300)
            p = [kb_prefix+ind for ind in examples['positive examples']]
            n = [kb_prefix+ind for ind in examples['negative examples']]
            print('Target concept: ', str_target_concept)
            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, p)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, n)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
            t0 = time.time()
            model.fit(lp, verbose=args.verbose)
            t1 = time.time()
            duration = t1-t0
            for desc in model.best_hypotheses(1):
                target_expression = dl_parser.parse_expression(str_target_concept) # The target class expression
                positive_examples = set(kb.individuals(target_expression))
                negative_examples = all_individuals-positive_examples
                acc, f1 = evaluator.evaluate(desc.concept, positive_examples, negative_examples)
                print(f"*** Acc: {acc}, F1: {f1} ***")
                Result_dict['F-measure'].append(f1)
                Result_dict['Accuracy'].append(acc)
                Result_dict['Runtime'].append(duration)
                Result_dict['Prediction'].append(dlsr.render(desc.concept))
                Result_dict['Learned Concept'].append(str_target_concept)
            
        for key in Result_dict:
            if key in ['Prediction', 'Learned Concept']: continue
            Avg_result[key]['mean'] = np.mean(Result_dict[key])
            Avg_result[key]['std'] = np.std(Result_dict[key])

        if not os.path.exists(f"../datasets/{kb_name}/Results/"):
            os.mkdir(f"../datasets/{kb_name}/Results/")

        if not os.path.exists(f"../datasets/{kb_name}/Results/"):
            os.mkdir(f"../datasets/{kb_name}/Results/")

        with open(f'../datasets/{kb_name}/Results/EvoLearner_results.json', 'w') as file_descriptor1:
                    json.dump(Result_dict, file_descriptor1, ensure_ascii=False, indent=3)

        with open(f'../datasets/{kb_name}/Results/EvoLearner_avg_results.json', 'w') as file_descriptor2:
                    json.dump(Avg_result, file_descriptor2, indent=3)

        print("\nnAvg results: ", Avg_result)
        print()