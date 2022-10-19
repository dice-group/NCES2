import os
import random, json
from ontolearn.knowledge_base import KnowledgeBase
from tqdm import tqdm
base_path = os.path.dirname(os.path.realpath(__file__)).split('ecii')[0]

def create_config(kb):
    print(f"\n## {kb.upper()} KB ##\n")
    example_config = 'airport_terminal_one_vs_all.config'
    with open(example_config) as example_file:
        example = example_file.readlines()
    lp_path = base_path+'datasets/'+kb+'/Test_data/Data.json'
    with open(lp_path,"r") as lp_file:
        data_test = json.load(lp_file)
    data_test = list(data_test.items())
    kb_file = base_path+'datasets/'+kb+'/'+kb+'.owl'
    KB = KnowledgeBase(path=kb_file)
    prefix = list(KB.individuals())[0].get_iri().as_str().split('#')[0]+'#'
    if not os.path.exists(kb):
        os.mkdir(kb)
    ks_file = kb_file.split('/')[-1]
    for i, (_, examples) in tqdm(enumerate(data_test), desc='writing problems...'):
        p = ['\"'+'ex:'+ind.split('#')[-1]+'\"' for ind in examples['positive examples']] # encode with urllib as required by dllearner ontology manager
        n = ['\"'+'ex:'+ind.split('#')[-1]+'\"' for ind in examples['negative examples']]
        p = '{ '+','.join(p)+' }'
        n = '{ '+','.join(n)+' }'
        with open(kb+'/'+kb+f'_{i+1}.config', 'w') as file:
            for line in example:
                if 'namespace=' in line:
                    file.write(f'namespace={prefix}\n')
                elif 'ks.fileName' in line:
                    file.write(f'ks.fileName = "{ks_file}"\n')
                elif 'prefixes =' in line:
                    file.write(f'prefixes = [ ("ex","{prefix}") ]\n')
                elif 'lp.positiveExamples' in line:
                    file.write(f'lp.positiveExamples = {p}\n')
                elif 'lp.negativeExamples' in line:
                    file.write(f'lp.negativeExamples = {n}\n')
                elif 'objectProperties' in line:
                    continue
                else:
                    file.write(line)
                    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kbs', type=str, nargs='+', default=['semantic_bible', 'mutagenesis', 'carcinogenesis', 'vicodi'])
    for kb in parser.parse_args().kbs:
        create_config(kb)