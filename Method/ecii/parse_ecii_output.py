import os, json
import re, numpy as np

base_path = os.path.dirname(os.path.realpath(__file__)).split('ecii')[0]
def parse_output(kb):
    files = sorted([f for f in os.listdir(kb) if f.endswith('.txt')], key=lambda x: float(x.split('_')[-7].split('.')[0]))
    results = {'F-measure': [], 'Runtime': []}
    for file in files:
        with open(f'{kb}/{file}') as f:
            output = f.readlines()
        for line in output:
            if 'duration' in line:
                time = re.findall(r'[+-]?\d+\.?\d+', line)[0]
                time = float(time)
                results['Runtime'].append(time)
            if 'f_measure' in line:
                f1 = re.findall(r'[+-]?\d+\.?\d+', line)[0]
                f1 = float(f1)
                results['F-measure'].append(f1)
                break
        
    with open(base_path+f'/Datasets/{kb}/Results/concept_learning_results_ecii.json', 'w') as ecii_file:
        json.dump(results, ecii_file, indent=3)
        
    avg_res = dict([(key, {'mean': np.mean(results[key]), 'std': np.std(results[key])}) for key in results])
    
    with open(base_path+f'/Datasets/{kb}/Results/concept_learning_avg_results__ecii.json', 'w') as ecii_file:
        json.dump(avg_res, ecii_file, indent=3)
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', type=str, nargs='+', default=['family-benchmark', 'semantic_bible', 'mutagenesis', 'carcinogenesis', 'vicodi'])
    args = parser.parse_args()
    for kb in args.kb:
        print(f'Parsing ECCI output on {kb}\n')
        parse_output(kb)
    print(f'\nDone parsing ECCI output on the following KBs: {args.kb}')
    
    
    