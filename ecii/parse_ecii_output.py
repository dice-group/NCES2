import os, json, sys
import re, numpy as np
from collections import defaultdict
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from ontolearn.knowledge_base import KnowledgeBase
base_path = os.path.dirname(os.path.realpath(__file__)).split('ecii')[0]
sys.path.append(base_path)
from utils.evaluator import Evaluator

def parse_output(kb):
    with open(base_path+f"datasets/{kb}/Test_data/Data.json","r") as lp_file:
        data_test = json.load(lp_file)
    files = sorted([f for f in os.listdir(kb) if f.endswith('.txt')], key=lambda x: float(x.split('_')[-7].split('.')[0]))
    Result_dict = {'F-measure': [], 'Accuracy': [], 'Runtime': [], 'Prediction': [], 'Learned Concept': []} #{'F-measure': [], 'Runtime': []}
    Avg_result = defaultdict(lambda: defaultdict(float))
    KB = KnowledgeBase(path=base_path+f"datasets/{kb}/{kb}.owl")
    all_individuals = set(KB.individuals())
    kb_namespace = list(KB.individuals())[0].get_iri().get_namespace()
    evaluator = Evaluator(KB)
    dl_parser = DLSyntaxParser(namespace = kb_namespace)
    dlsr = DLSyntaxObjectRenderer()
    for file, (str_target_concept, examples) in zip(files, data_test):
        with open(f'{kb}/{file}') as f:
            output = f.readlines()
        for line in output:
            if 'duration' in line:
                time = re.findall(r'[+-]?\d+\.?\d+', line)[0]
                time = float(time)
            if 'solution 1' in line:
                solution = line.split(":")[-1].strip()
                break
        target_expression = dl_parser.parse(str_target_concept) # The target class expression
        prediction = dl_parser.parse(solution)
        positive_examples = set(KB.individuals(target_expression))
        negative_examples = all_individuals-positive_examples
        acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
        print(f"*** Acc: {acc}, F1: {f1} ***")
        Result_dict['F-measure'].append(f1)
        Result_dict['Accuracy'].append(acc)
        Result_dict['Runtime'].append(time)
        Result_dict['Prediction'].append(dlsr.render(prediction))
        Result_dict['Learned Concept'].append(str_target_concept)
    for key in Result_dict:
        if key in ['Prediction', 'Learned Concept']: continue
        Avg_result[key]['mean'] = np.mean(Result_dict[key])
        Avg_result[key]['std'] = np.std(Result_dict[key])

    if not os.path.exists(base_path+f"datasets/{kb}/Results/"):
        os.mkdir(base_path+f"datasets/{kb}/Results/")

    if not os.path.exists(base_path+f"datasets/{kb}/Results/"):
        os.mkdir(currentpath.split("dllearner")[0]+f"datasets/{kb}/Results/")

    with open(base_path+f'datasets/{kb}/Results/ECII_results.json', 'w') as file_descriptor1:
                json.dump(Result_dict, file_descriptor1, ensure_ascii=False, indent=3)

    with open(base_path+f'datasets/{kb}/Results/ECII_avg_results.json', 'w') as file_descriptor2:
                json.dump(Avg_result, file_descriptor2, indent=3)

    print("Avg results: ", Avg_result)
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kbs', type=str, nargs='+', default=['semantic_bible', 'mutagenesis', 'carcinogenesis', 'vicodi'])
    args = parser.parse_args()
    for kb in args.kbs:
        print(f'Parsing ECCI output on {kb}\n')
        parse_output(kb)
    print(f'\nDone parsing ECCI output on the following KBs: {args.kbs}')
    
    
    