import json, time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os, sys
from sklearn.model_selection import train_test_split

import urllib.parse

currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentpath.split("dllearner")[0])

from binders import DLLearnerBinder
from utils.concept_length import concept_length
from ontolearn.knowledge_base import KnowledgeBase

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, nargs='+', default=['celoe', 'eltl'], help='The algorithm to be used')
    parser.add_argument('--kbs', type=str, nargs='+', default=['semantic_bible', 'mutagenesis', 'carcinogenesis', 'vicodi'], \
                       help='The knowledge bases of interest')
    parser.add_argument('--max_runtime', type=int, default=300, help='The maximum runtime of CELOE')
    args = parser.parse_args()
    
    for kb in args.kbs:
        lp_path = currentpath.split("dllearner")[0]+f"datasets/{kb}/Test_data/Data.json"
        with open(lp_path,"r") as lp_file:
            data_test = json.load(lp_file)
        data_test = list(data_test.items())

        kb_path = currentpath.split("dllearner")[0]+f'datasets/{kb}/{kb}.owl'
        # To download DL-learner,  https://github.com/SmartDataAnalytics/DL-Learner/releases.
        dl_learner_binary_path = currentpath.split("dllearner")[0]+'dllearner-1.4.0/'
        knowledge_base = KnowledgeBase(path=kb_path)
        kb_namespace = list(knowledge_base.individuals())[0].get_iri().get_namespace()
        kb_prefix = kb_namespace[:kb_namespace.rfind("/")+1]

        for model in args.algo:
            algo = DLLearnerBinder(binary_path=dl_learner_binary_path, kb_path=kb_path, model=model)
            #{'F-measure': [], 'Prediction': [], 'Learned Concept': [], 'Runtime': []} 
            Result_dict = {'F-measure': [], 'Accuracy': [], 'Runtime': [], 'Prediction': [], 'Length': [], 'Learned Concept': []}
            Avg_result = defaultdict(lambda: defaultdict(float))
            print("#"*60)
            print(f"{model.upper()} on "+kb_path.split("/")[-2]+" knowledge base")
            print("#"*60)
            for str_target_concept, examples in tqdm(data_test, desc=f'Learning {len(data_test)} problems'):
                print('TARGET CONCEPT:', str_target_concept)
                p = [urllib.parse.quote(kb_prefix+ind) for ind in examples['positive examples']] # encode with urllib as required by dllearner ontology manager
                n = [urllib.parse.quote(kb_prefix+ind) for ind in examples['negative examples']]
                t0 = time.time()
                try:
                    best_pred_algo = algo.fit(pos=p, neg=n, max_runtime=args.max_runtime).best_hypotheses() # Start learning
                except Exception as err:
                    print("\n",err)
                    best_pred_algo = {'Model': model, 'Prediction': '', 'Accuracy': 0.0, 'F-measure': 0.0}
                t1 = time.time()
                duration = t1-t0
                print('Best prediction: ', best_pred_algo)
                print()
                if model == 'ocel': # No F-measure for OCEL
                    Result_dict['F-measure'].append(-100.)
                else:
                    Result_dict['F-measure'].append(best_pred_algo['F-measure'])
                Result_dict['Accuracy'].append(best_pred_algo['Accuracy'])
                if not 'Runtime' in best_pred_algo or best_pred_algo['Runtime'] is None:
                    Result_dict['Runtime'].append(duration)
                else:
                    Result_dict['Runtime'].append(best_pred_algo['Runtime'])
                if best_pred_algo['Prediction'] is None:
                    Result_dict['Prediction'].append('None')
                    Result_dict['Length'].append(10)
                    Result_dict['Learned Concept'].append(str_target_concept)
                else:
                    Result_dict['Prediction'].append(best_pred_algo['Prediction'])
                    Result_dict['Length'].append(concept_length(best_pred_algo['Prediction']))
                    Result_dict['Learned Concept'].append(str_target_concept)

            for key in Result_dict:
                if key in ['Prediction', 'Learned Concept']: continue
                Avg_result[key]['mean'] = np.mean(Result_dict[key])
                Avg_result[key]['std'] = np.std(Result_dict[key])

            if not os.path.exists(currentpath.split("dllearner")[0]+f"datasets/{kb}/Results/"):
                os.mkdir(currentpath.split("dllearner")[0]+f"datasets/{kb}/Results/")

            if not os.path.exists(currentpath.split("dllearner")[0]+f"datasets/{kb}/Results/"):
                os.mkdir(currentpath.split("dllearner")[0]+f"datasets/{kb}/Results/")

            with open(currentpath.split("dllearner")[0]+f'datasets/{kb}/Results/concept_learning_results_'+model+'.json', 'w') as file_descriptor1:
                        json.dump(Result_dict, file_descriptor1, ensure_ascii=False, indent=3)

            with open(currentpath.split("dllearner")[0]+f'datasets/{kb}/Results/concept_learning_avg_results__'+model+'.json', 'w') as file_descriptor2:
                        json.dump(Avg_result, file_descriptor2, indent=3)

            print("Avg results: ", Avg_result)
            print()
