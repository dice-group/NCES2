from argparse import Namespace
import argparse
from helper_reproduce_nces1 import evaluate_nces
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    with open("settings.json") as setting:
        nces_args = json.load(setting)
    nces_args = Namespace(**nces_args)

    parser = argparse.ArgumentParser()

    parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], choices=['carcinogenesis', 'mutagenesis', 'semantic_bible', 'vicodi'], help='Knowledge base name')
    parser.add_argument('--kb_emb_model', type=str, default="ConEx", help='KB embedding model')
    parser.add_argument('--ensemble', type=str2bool, default=True, help='Whether to also evaluate ensemble models')
    parser.add_argument('--save_results', type=str2bool, default=False, help='Whether to save the evaluation results')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Whether to print the target and predicted class expressions')
    args = parser.parse_args()
    
    nces_args.kb_emb_model = args.kb_emb_model
    for kb in args.kbs:
        print()
        evaluate_nces(kb_name=kb, model_name='SetTransformer', args=nces_args, save_results=args.save_results, verbose=args.verbose)