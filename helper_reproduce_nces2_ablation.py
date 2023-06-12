import os, random
from utils.simple_solution import SimpleSolution
from utils.evaluator import Evaluator
from utils.data import Data
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from nces2 import BaseConceptSynthesis
from nces2.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from dataloader import NCESDataLoaderInference
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json
import torch
import numpy as np, time
from collections import defaultdict

def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp


num_examples = 1000
def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    for pos_emb, neg_emb in batch:
        if pos_emb.ndim != 2:
            pos_emb = pos_emb.reshape(1, -1)
        if neg_emb.ndim != 2:
            neg_emb = neg_emb.reshape(1, -1)
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
    pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, num_examples - pos_emb_list[0].shape[0]), "constant", 0)
    pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, num_examples - neg_emb_list[0].shape[0]), "constant", 0)
    neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    return pos_emb_list, neg_emb_list

def predict(kb, model, embedding_model, args):
    data_test_path = f"datasets/{kb}/Test_data/Data.json"
    with open(data_test_path, "r") as file:
        test_data = json.load(file)
    args.path_to_triples = f"datasets/{kb}/Triples/"
    global num_examples
    if isinstance(models, list):
        num_examples = models[0].num_examples
        vocab = models[0].vocab
        inv_vocab = models[0].inv_vocab
    else:
        num_examples = models.num_examples
        vocab = models.vocab
        inv_vocab = models.inv_vocab
    kb_embedding_data = Data(args)
    test_dataset = NCESDataLoaderInference(test_data, kb_embedding_data)
    preds = []
    models = models.eval()
    test_dataset.load_embeddings(embedding_model.eval())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
    for x1, x2 in tqdm(test_dataloader):
        pred_sequence, _ = models(x1, x2)
        preds.append(pred_sequence)
    return [item[0] for item in test_data], np.concatenate(preds, 0)


def initialize_synthesizer(num_inds, args):
    args.num_inds = num_inds
    nces2 = ConceptSynthesizer(args)
    nces2.refresh()
    return nces2.model, nces2.embedding_model


def synthesize_class_expressions(kb, num_inds, ablation_type, args):
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    embs = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_{ablation_type}_inducing_points32.pt", map_location = torch.device("cpu"))
    setattr(args, 'num_entities', embs['emb_ent_real.weight'].shape[0])
    setattr(args, 'num_relations', embs['emb_rel_real.weight'].shape[0])
    if not hasattr(args, 'ablation'):
        setattr(args, 'ablation', ablation_type)
    else:
        args.ablation = ablation_type
    model, embedding_model = initialize_synthesizer(num_inds, args)
    model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_{ablation_type}_inducing_points{num_inds}.pt",
                       map_location=torch.device("cpu")))
    embedding_model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_{ablation_type}_inducing_points{num_inds}.pt",
                                    map_location = torch.device("cpu")))
    return predict(kb, model, embedding_model, args)
    

def evaluate_nces(kb_name, num_inds, ablation_type, args, save_results=False, verbose=False):
    print('#'*50)
    print('NCES2 evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    desc = f"{ablation_type}"
    all_metrics = {m: defaultdict(lambda: defaultdict(list)) for m in [f"SetTransformer_I{inds}" for inds in num_inds]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = list(kb.individuals())[0].get_iri().get_namespace()
    print("KB namespace: ", namespace)
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    for model_name in all_metrics.keys():
        num_inds = int(model_name.split("I")[-1])
        t0 = time.time()
        targets, predictions = synthesize_class_expressions(kb_name, num_inds, ablation_type, args)
        t1 = time.time()
        duration = (t1-t0)/len(predictions)
        for i, pb_str in enumerate(targets):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = -1
            pred = predictions[i][:end_idx]
            try:
                prediction = dl_parser.parse("".join(pred.tolist()))
            except Exception:
                try:
                    pred = simpleSolution.predict(predictions[i].sum())
                    prediction = dl_parser.parse(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
            if prediction is None:
                prediction = dl_parser.parse('⊤')
            target_expression = dl_parser.parse(pb_str) # The target class expression
            positive_examples = set(kb.individuals(target_expression))
            negative_examples = all_individuals-positive_examples
            try:
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            except Exception as err:
                print("Parsing error on ", prediction)
                print(err)
                prediction = dl_parser.parse('⊤')
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            if verbose:
                print(f'Problem {i}, Target: {pb_str}, Prediction: {simpleSolution.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
                print()
            all_metrics[model_name]['acc']['values'].append(acc)
            try:
                all_metrics[model_name]['prediction']['values'].append(simpleSolution.renderer.render(prediction))
            except:
                print(f"Error in rendering {prediction}")
                all_metrics[model_name]['prediction']['values'].append("Unknown")
            all_metrics[model_name]['f1']['values'].append(f1)
            all_metrics[model_name]['time']['values'].append(duration)
            
        for metric in all_metrics[model_name]:
            if metric != 'prediction':
                all_metrics[model_name][metric]['mean'] = [np.mean(all_metrics[model_name][metric]['values'])]
                all_metrics[model_name][metric]['std'] = [np.std(all_metrics[model_name][metric]['values'])]
        
        print(model_name+' Speed: {}s +- {} / lp'.format(round(all_metrics[model_name]['time']['mean'][0], 2),\
                                                               round(all_metrics[model_name]['time']['std'][0], 2)))
        print(model_name+' Avg Acc: {}% +- {} / lp'.format(round(all_metrics[model_name]['acc']['mean'][0], 2),\
                                                               round(all_metrics[model_name]['acc']['std'][0], 2)))
        print(model_name+' Avg F1: {}% +- {} / lp'.format(round(all_metrics[model_name]['f1']['mean'][0], 2),\
                                                               round(all_metrics[model_name]['f1']['std'][0], 2)))
        print()
        if save_results:
            with open("datasets/"+kb_name+f"/Results/NCES2_{args.kb_emb_model}"+desc+".json", "w") as file:
                json.dump(all_metrics, file, indent=3, ensure_ascii=False)
