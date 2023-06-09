import os, random
from utils.evaluator import Evaluator
from ontolearn.knowledge_base import KnowledgeBase
#from nces1.utils.base import DataLoaderBase
from utils.simple_solution import SimpleSolution
from nces1.nces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from nces1.utils.dataloader import NCESDataLoaderInference
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json
import torch, pandas as pd
import numpy as np, time
from collections import defaultdict

def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp


def get_data(kb, embeddings, kwargs):
    test_data_path = f"datasets/{kb}/Test_data/Data.json"
    with open(test_data_path, "r") as file:
        test_data = json.load(file)
    test_dataset = NCESDataLoaderInference(test_data, embeddings, kwargs)
    num_examples = test_dataset.num_examples
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
    print("Number of learning problems: ", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=kwargs.batch_size, num_workers=kwargs.num_workers, collate_fn=collate_batch, shuffle=False)
    return [item[0] for item in test_data], test_dataloader

def predict_class_expressions(kb, model_name, args):
    assert isinstance(model_name, str)
    print(f"\n## Single model prediction ({model_name})")
    model = torch.load(f"nces1/datasets/{kb}/Model_weights/{args.kb_emb_model}_{model_name}.pt", map_location=torch.device('cpu'))
    model.eval()
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    embeddings = pd.read_csv(f"nces1/datasets/{kb}/Embeddings/{args.kb_emb_model}_entity_embeddings.csv").set_index('Unnamed: 0')
    test_data, dataloader = get_data(kb, embeddings, args)
    preds = []
    targets = []
    i = 0
    for x1, x2 in tqdm(dataloader):
        pred_sequence, _ = model(x1, x2)
        preds.append(pred_sequence)
        targets.extend(test_data[i:i+x1.shape[0]])
        i += x1.shape[0]
    return np.concatenate(preds, 0), targets


def evaluate_nces(kb_name, model_name, args, save_results=False, verbose=False):
    print('#'*50)
    print('NCES evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    all_metrics = {model_name: defaultdict(lambda: defaultdict(list))}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    kb_namespace = list(kb.individuals())[0].get_iri().get_namespace()
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = kb_namespace)
    all_individuals = set(kb.individuals())
    t0 = time.time()
    predictions, targets = predict_class_expressions(kb_name, model_name, args)
    t1 = time.time()
    duration = (t1-t0)/len(predictions)
    for i, pb_str in enumerate(targets):
        pb_str = "".join(before_pad(pb_str))
        try:
            end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
        except IndexError:
            end_idx = 1
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
        try:
            positive_examples = set(kb.individuals(target_expression))
            negative_examples = all_individuals-positive_examples
            acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
        except Exception as err:
            print(err)
        if verbose:
            print(f'Problem {i}, Target: {pb_str}, Prediction: {simpleSolution.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
            print()
        all_metrics[model_name]['acc']['values'].append(acc)
        all_metrics[model_name]['prediction']['values'].append(simpleSolution.renderer.render(prediction))
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
        with open("datasets/"+kb_name+f"/Results/NCES1_{args.kb_emb_model}.json", "w") as file:
            json.dump(all_metrics, file, ensure_ascii=False)