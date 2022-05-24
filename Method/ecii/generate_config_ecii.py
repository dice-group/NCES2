import os
import random, json
from sklearn.model_selection import train_test_split
from ontolearn import KnowledgeBase
from tqdm import tqdm
base_path = os.path.dirname(os.path.realpath(__file__)).split('ecii')[0]

def create_config(kb):
    example_config = 'airport_terminal_one_vs_all.config'
    with open(example_config) as example_file:
        example = example_file.readlines()
    lp_path = base_path+'Datasets/'+kb+'/Train_data/Data.json'
    with open(lp_path,"r") as lp_file:
        data = json.load(lp_file)
    data = list(data.items())
    _, data_test = train_test_split(data, test_size=0.2, random_state=123)
    random.seed(142)
    data_test = random.sample(data_test, 200)
    kb_file = base_path+'Datasets/'+kb+'/'+kb+'.owl' if kb != 'family-benchmark' else base_path+'Datasets/'+kb+'/'+kb+'_rich_background.owl'
    KB = KnowledgeBase(path=kb_file)
    all_inds = {'\"'+'ex:'+ind.get_iri().as_str().split('/')[-1].split('#')[-1]+'\"' for ind in KB.individuals()}
    prefix = list(KB.individuals())[0].get_iri().as_str().split('#')[0]+'#'
    if not os.path.exists(kb):
        os.mkdir(kb)
    ks_file = kb_file.split('/')[-1]
    for i, examples in tqdm(enumerate(data_test), desc='writing problems...'):
        p = ['\"'+'ex:'+ind.split('#')[-1]+'\"' for ind in examples[1]['positive examples']]
        n = list(all_inds-set(p))
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
    parser.add_argument('--kb', type=str, nargs='+', default=['family-benchmark', 'semantic_bible', 'mutagenesis', 'carcinogenesis', 'vicodi'])
    for kb in parser.parse_args().kb:
        create_config(kb)