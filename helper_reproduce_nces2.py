import os, random
from utils.simple_solution import SimpleSolution
from utils.evaluator import Evaluator
from utils.data import Data
from ontolearn.knowledge_base import KnowledgeBase
from nces import BaseConceptSynthesis
from nces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from dataloader import NCESDataLoader
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

def compute_accuracy(prediction, target):
    def soft(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = set(before_pad(BaseConceptSynthesis.decompose(arg1_)))
        else:
            arg1_ = set(before_pad(arg1_))
        if isinstance(arg2_, str):
            arg2_ = set(before_pad(BaseConceptSynthesis.decompose(arg2_)))
        else:
            arg2_ = set(before_pad(arg2_))
        return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

    def hard(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = before_pad(BaseConceptSynthesis.decompose(arg1_))
        else:
            arg1_ = before_pad(arg1_)
        if isinstance(arg2_, str):
            arg2_ = before_pad(BaseConceptSynthesis.decompose(arg2_))
        else:
            arg2_ = before_pad(arg2_)
        return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
    soft_acc = sum(map(soft, prediction, target))/len(target)
    hard_acc = sum(map(hard, prediction, target))/len(target)
    return soft_acc, hard_acc

num_examples = 1000
def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    target_tokens_list = []
    target_labels = []
    for pos_emb, neg_emb, label in batch:
        if pos_emb.ndim != 2:
            pos_emb = pos_emb.reshape(1, -1)
        if neg_emb.ndim != 2:
            neg_emb = neg_emb.reshape(1, -1)
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
        target_labels.append(label)
    pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, num_examples - pos_emb_list[0].shape[0]), "constant", 0)
    pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, num_examples - neg_emb_list[0].shape[0]), "constant", 0)
    neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
    return pos_emb_list, neg_emb_list, target_labels

def predict(kb, models, args):
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
    test_dataset = NCESDataLoader(test_data, kb_embedding_data, vocab, inv_vocab, args)
    soft_acc, hard_acc = 0.0, 0.0
    preds = []
    targets = []
    if isinstance(models, list):
        for i, model in enumerate(models):
            model = model.eval()
            scores = []
            num_inds = model.num_inds
            embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                        map_location = torch.device("cpu"))
            test_dataset.load_embeddings(embedding_model.eval())
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
            for x1, x2, labels in tqdm(test_dataloader):
                if i == 0:
                    target_sequence = model.inv_vocab[labels]
                    targets.append(target_sequence) # The target sequence does not depend on the current model
                _, sc = model(x1, x2)
                scores.append(sc.detach()) 
            scores = torch.cat(scores, 0)
            if i == 0:
                Scores = scores
            else:
                Scores = Scores + scores
        Scores = Scores / len(models)
        pred_sequence = model.inv_vocab[Scores.argmax(1)]
        targets = np.concatenate(targets, 0)
        assert len(targets) == len(pred_sequence), f"Something went wrong: len(targets) is {len(targets)} and len(predictions) is {len(pred_sequence)}"
        soft_acc, hard_acc = compute_accuracy(pred_sequence, targets)
        print(f"Average syntactic accuracy, Soft: {soft_acc}%, Hard: {hard_acc}%")
        return pred_sequence, targets
    else:
        models = models.eval()
        num_inds = models.num_inds
        embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                    map_location = torch.device("cpu"))
        test_dataset.load_embeddings(embedding_model.eval())
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
        for x1, x2, labels in tqdm(test_dataloader):
            target_sequence = models.inv_vocab[labels]
            pred_sequence, _ = models(x1, x2)
            preds.append(pred_sequence)
            targets.append(target_sequence)
            s_acc, h_acc = compute_accuracy(pred_sequence, target_sequence)
            soft_acc += s_acc
            hard_acc += h_acc
        print(f"Average syntactic accuracy, Soft: {soft_acc/len(test_dataloader)}%, Hard: {hard_acc/len(test_dataloader)}%")
        return np.concatenate(preds, 0), np.concatenate(targets, 0)

def synthesize_class_expressions(kb, num_inds, args):
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    if isinstance(num_inds, list):
        print(f"\n## Ensemble prediction ({'+'.join([f'SetTransformer_I{inds}' for inds in num_inds])})")
    
        models = [torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{inds}.pt",
                             map_location=torch.device("cpu")) for inds in num_inds]
        return predict(kb, models, args)
    else:
        print(f"\n## Single model prediction (SetTransformer_I{num_inds})")
        model = torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{num_inds}.pt",
                           map_location=torch.device("cpu"))
        return predict(kb, model, args)
    
    

def evaluate_nces(kb_name, num_inds, args, save_results=False, verbose=False):
    print('#'*50)
    print('NCES2 evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    desc = ""
    all_metrics = {m: defaultdict(lambda: defaultdict(list)) for m in [f"SetTransformer_I{inds}" for inds in num_inds]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    for model_name in all_metrics.keys():
        num_inds = int(model_name.split("I")[-1])
        t0 = time.time()
        predictions, targets = synthesize_class_expressions(kb_name, num_inds, args)
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
            with open("datasets/"+kb_name+f"/Results/NCES_{args.kb_emb_model}"+desc+".json", "w") as file:
                json.dump(all_metrics, file, indent=3, ensure_ascii=False)

                
def evaluate_ensemble(kb_name, args, save_results=False, verbose=False):
    print('#'*50)
    print('NCES2 evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    all_metrics = {'+'.join(combine): defaultdict(lambda: defaultdict(list)) for combine in [["SetTransformer_I32", "SetTransformer_I64"], \
                                        ["SetTransformer_I32", "SetTransformer_I128"], ["SetTransformer_I64", "SetTransformer_I128"],\
                                        ["SetTransformer_I32", "SetTransformer_I64", "SetTransformer_I128"]]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    for combine in all_metrics.keys():     
        t0 = time.time()
        num_inds = [int(model_name.split("I")[-1]) for model_name in combine.split("+")]
        predictions, targets = synthesize_class_expressions(kb_name, num_inds, args)
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
            all_metrics[combine]['acc']['values'].append(acc)
            try:
                all_metrics[combine]['prediction']['values'].append(simpleSolution.renderer.render(prediction))
            except:
                all_metrics[combine]['prediction']['values'].append("Unknown")
            all_metrics[combine]['f1']['values'].append(f1)
            all_metrics[combine]['time']['values'].append(duration)

        for metric in all_metrics[combine]:
            if metric != 'prediction':
                all_metrics[combine][metric]['mean'] = [np.mean(all_metrics[combine][metric]['values'])]
                all_metrics[combine][metric]['std'] = [np.std(all_metrics[combine][metric]['values'])]

        print(combine+' Speed: {}s +- {} / lp'.format(round(all_metrics[combine]['time']['mean'][0], 2),\
                                                               round(all_metrics[combine]['time']['std'][0], 2)))
        print(combine+' Avg Acc: {}% +- {} / lp'.format(round(all_metrics[combine]['acc']['mean'][0], 2),\
                                                               round(all_metrics[combine]['acc']['std'][0], 2)))
        print(combine+' Avg F1: {}% +- {} / lp'.format(round(all_metrics[combine]['f1']['mean'][0], 2),\
                                                               round(all_metrics[combine]['f1']['std'][0], 2)))

        print()

    if save_results:
        with open(f"datasets/{kb_name}/Results/NCES_{args.kb_emb_model}_Ensemble.json", "w") as file:
            json.dump(all_metrics, file, indent=3, ensure_ascii=False)
