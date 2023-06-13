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
import re

def build_nces2_vocabulary(data_train, data_test, kb, args):
    def add_data_values(path):
        print("\n*** Finding relevant data values ***")
        values = set()
        for ce, lp in data_train+data_test:
            if '[' in ce:
                for val in re.findall("\[(.*?)\]", ce):
                    values.add(val.split(' ')[-1])
        print("*** Done! ***\n")
        print("Added values: ", values)
        print()
        return list(values)
    renderer = DLSyntaxObjectRenderer()
    individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
    atomic_concepts = list(kb.ontology().classes_in_signature())
    atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
    role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                 [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
    vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                'double', 'integer', 'date', 'xsd']
    quantified_restriction_values = [str(i) for i in range(1,12)]
    data_values = add_data_values(args.knowledge_base_path)
    vocab = vocab + data_values + quantified_restriction_values
    vocab = sorted(set(vocab)) + ['PAD']
    print("Vocabulary size: ", len(vocab))
    num_examples = min(args.num_examples, kb.individuals_count()//2)
    return vocab, num_examples

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

def predict(kb, test_data, model, embedding_model, args):
    args.path_to_triples = f"datasets/{kb}/Triples/"
    global num_examples
    num_examples = model.num_examples
    vocab = model.vocab
    inv_vocab = model.inv_vocab
    kb_embedding_data = Data(args)
    test_dataset = NCESDataLoaderInference(test_data, kb_embedding_data)
    preds = []
    model = model.eval()
    test_dataset.load_embeddings(embedding_model.eval())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
    for x1, x2 in tqdm(test_dataloader):
        pred_sequence, _ = model(x1, x2)
        preds.append(pred_sequence)
    return [item[0] for item in test_data], np.concatenate(preds, 0)


def initialize_synthesizer(vocab, num_examples, num_inds, args):
    args.num_inds = num_inds
    nces2 = ConceptSynthesizer(vocab, num_examples, args)
    nces2.refresh()
    return nces2.model, nces2.embedding_model


def synthesize_class_expressions(kb, test_data, vocab, num_examples, num_inds, ablation_type, args):
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    embs = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_{ablation_type}_inducing_points32.pt", map_location = torch.device("cpu"))
    setattr(args, 'num_entities', embs['emb_ent_real.weight'].shape[0])
    setattr(args, 'num_relations', embs['emb_rel_real.weight'].shape[0])
    if not hasattr(args, 'ablation'):
        setattr(args, 'ablation', ablation_type)
    else:
        args.ablation = ablation_type
    model, embedding_model = initialize_synthesizer(vocab, num_examples, num_inds, args)
    model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_{ablation_type}_inducing_points{num_inds}.pt",
                       map_location=torch.device("cpu")))
    embedding_model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_{ablation_type}_inducing_points{num_inds}.pt",
                                    map_location = torch.device("cpu")))
    return predict(kb, test_data, model, embedding_model, args)
    

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
    with open(f"datasets/{kb_name}/Test_data/Data.json", "r") as file:
        test_data = json.load(file)
    with open(f"datasets/{kb_name}/Train_data/Data_{ablation_type}.json", "r") as file:
        train_data = json.load(file)
    vocab, num_examples = build_nces2_vocabulary(train_data, [], kb, args)
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    for model_name in all_metrics.keys():
        num_inds = int(model_name.split("I")[-1])
        t0 = time.time()
        targets, predictions = synthesize_class_expressions(kb_name, test_data, vocab, num_examples, num_inds, ablation_type, args)
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
