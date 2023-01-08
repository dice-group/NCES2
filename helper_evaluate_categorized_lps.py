import os, random
from utils.syntax_checker import SyntaxChecker
from utils.evaluator import Evaluator
from ontolearn.knowledge_base import KnowledgeBase
from nces import BaseConceptSynthesis
from nces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from dataloader import CSDataLoaderInference
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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


def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    for pos_emb, neg_emb in batch:
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
    try:
        pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    except:
        temp_pos = []
        for t in pos_emb_list:
            if t.ndim != 2:
                temp_pos.append(t.reshape(1, -1))
            else:
                temp_pos.append(t)
        pos_emb_list = pad_sequence(temp_pos, batch_first=True, padding_value=0)

    try:
        neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    except:
        temp_neg = []
        for t in neg_emb_list:
            if t.ndim != 2:
                temp_neg.append(t.reshape(1, -1))
            else:
                temp_neg.append(t)
        neg_emb_list = pad_sequence(temp_neg, batch_first=True, padding_value=0)
    return pos_emb_list, neg_emb_list

def predict(kb, models, args):
    test_data = []
    for lp_type in ['real_values', 'boolean_values', 'inverse_property_values', 'cardinality_restrictions']:
        if not os.path.isfile(f"complex_lps/{kb}/{lp_type}.json"): continue
        with open(f"complex_lps/{kb}/{lp_type}.json") as file:
            test_data += json.load(file)
    print("\n*** Number of learning problems: ", len(test_data), "***\n")
    args.path_to_triples = f"datasets/{kb}/Triples/"
    test_dataset = CSDataLoaderInference(test_data, args)
    preds = []
    targets = [ce[0] for ce in test_data]
    if isinstance(models, list):
        for i, model in enumerate(models):
            model.eval()
            scores = []
            num_inds = model.num_inds
            embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                        map_location = torch.device("cpu"))
            test_dataset.load_embeddings(embedding_model)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
            for x1, x2 in tqdm(test_dataloader):
                _, sc = model(x1, x2)
                scores.append(sc) 
            scores = torch.cat(scores, 0)
            if i == 0:
                Scores = scores
            else:
                Scores = Scores + scores
        Scores = Scores / len(models)
        pred_sequence = model.inv_vocab[Scores.argmax(1)]
        return pred_sequence, np.array(targets)
    else:
        models.eval()
        num_inds = models.num_inds
        embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                    map_location = torch.device("cpu"))
        test_dataset.load_embeddings(embedding_model)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
        for x1, x2 in tqdm(test_dataloader):
            pred_sequence, _ = models(x1, x2)
            preds.append(pred_sequence)
        return np.concatenate(preds, 0), np.array(targets)

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
    print('#'*100)
    print('NCES evaluation on complex learning problems for {} KB:'.format(kb_name))
    print('#'*100)
    All_metrics = {m: defaultdict(lambda: defaultdict(list)) for m in [f"SetTransformer_I{inds}" for inds in num_inds]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    syntax_checker = SyntaxChecker(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    All_individuals = set(kb.individuals())
    for model_name in All_metrics.keys():
        inds = int(model_name.split("I")[-1])
        t0 = time.time()
        predictions, targets = synthesize_class_expressions(kb_name, inds, args)
        t1 = time.time()
        duration = (t1-t0)/len(predictions)
        for i, pb_str in enumerate(targets):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = 1
            pred = predictions[i][:end_idx]
            #print("Before parsing: ", pred.sum())
            # In the following, try to repair an expression if one parenthesis is missing
            succeed = False
            if (pred=='(').sum() > (pred==')').sum():
                for i in range(len(pred))[::-1]:
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,')')))
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
                        
            elif (pred==')').sum() > (pred=='(').sum():
                for i in range(len(pred)):
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,'(')))
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
            else:
                try:
                    prediction = dl_parser.parse_expression("".join(pred.tolist()))
                except Exception:
                    try: # Try extracting the closest class expression with syntax checker
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
            if prediction is None:
                prediction = syntax_checker.get_concept(['⊤'])
            try:
                target_expression = dl_parser.parse_expression(pb_str) # The target class expression
            except:
                continue
            #try:
            positive_examples = {ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(target_expression)}
            negative_examples = All_individuals-positive_examples
            try:
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            except Exception as e:
                print("Error on ", prediction)
                print(e)
                prediction = syntax_checker.get_concept(['⊤'])
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
                
            if verbose:
                print(f'Problem {i}, Target: {pb_str}, Prediction: {syntax_checker.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
                print()
            All_metrics[model_name]['acc']['values'].append(acc)
            try:
                All_metrics[model_name]['prediction']['values'].append(syntax_checker.renderer.render(prediction))
            except:
                print(f"Error in rendering {prediction}")
                All_metrics[model_name]['prediction']['values'].append("Unknown")
            All_metrics[model_name]['f1']['values'].append(f1)
            All_metrics[model_name]['time']['values'].append(duration)
            
        

#        for metric in All_metrics[model_name]:
#            if metric != 'prediction':
#                All_metrics[model_name][metric]['mean'] = [np.mean(All_metrics[model_name][metric]['values'])]
#                All_metrics[model_name][metric]['std'] = [np.std(All_metrics[model_name][metric]['values'])]
#        
#        print(model_name+' Speed: {}s +- {} / lp'.format(round(All_metrics[model_name]['time']['mean'][0], 2),\
#                                                               round(All_metrics[model_name]['time']['std'][0], 2)))
#        print(model_name+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics[model_name]['acc']['mean'][0], 2),\
#                                                               round(All_metrics[model_name]['acc']['std'][0], 2)))
#        print(model_name+' Avg F1: {}% +- {} / lp'.format(round(All_metrics[model_name]['f1']['mean'][0], 2),\
#                                                               round(All_metrics[model_name]['f1']['std'][0], 2)))
#        print()
#        
    current_iter = 0
    for lp_type in ['real_values', 'boolean_values', 'inverse_property_values', 'cardinality_restrictions']:
        if not os.path.isfile(f"complex_lps/{kb_name}/{lp_type}.json"): continue
        all_metrics = {m: defaultdict(lambda: defaultdict(list)) for m in [f"SetTransformer_I{inds}" for inds in num_inds]}
        with open(f"complex_lps/{kb_name}/{lp_type}.json") as file:
            num_lps = len(json.load(file))
        for key1 in all_metrics:
            for key2 in All_metrics[key1]:
                for key3 in All_metrics[key1][key2]:
                    all_metrics[key1][key2][key3] = All_metrics[key1][key2][key3][current_iter:current_iter+num_lps]
                    if key2 != 'prediction':
                        all_metrics[key1][key2]['mean'] = [np.mean(all_metrics[key1][key2][key3])]
                        all_metrics[key1][key2]['std'] = [np.std(all_metrics[key1][key2][key3])]
            print(f'\n LPs of type {lp_type}')
            print(key1+' Speed: {}s +- {} / lp'.format(round(all_metrics[key1]['time']['mean'][0], 2),\
                                                                   round(all_metrics[key1]['time']['std'][0], 2)))
            print(key1+' Avg Acc: {}% +- {} / lp'.format(round(all_metrics[key1]['acc']['mean'][0], 2),\
                                                                   round(all_metrics[key1]['acc']['std'][0], 2)))
            print(key1+' Avg F1: {}% +- {} / lp'.format(round(all_metrics[key1]['f1']['mean'][0], 2),\
                                                        round(all_metrics[key1]['f1']['std'][0], 2)))
            print()
        current_iter += num_lps
        if save_results:
            with open("datasets/"+kb_name+f"/Results/NCES_{args.kb_emb_model}_{lp_type}.json", "w") as file:
                json.dump(all_metrics, file, indent=3, ensure_ascii=False)
    #if save_results:
    #    with open("datasets/"+kb_name+f"/Results/NCES_{args.kb_emb_model}_{lp_type}"+".json", "w") as file:
    #        json.dump(All_metrics, file, indent=3, ensure_ascii=False)

                
def evaluate_ensemble(kb_name, args, save_results=False, verbose=False):
    print('#'*100)
    print('NCES ensemble evaluation on complex learning problems of {} KB:'.format(kb_name))
    print('#'*100)
    All_metrics = {'+'.join(combine): defaultdict(lambda: defaultdict(list)) for combine in [["SetTransformer_I32", "SetTransformer_I64"], \
                                        ["SetTransformer_I32", "SetTransformer_I128"], ["SetTransformer_I64", "SetTransformer_I128"],\
                                        ["SetTransformer_I32", "SetTransformer_I64", "SetTransformer_I128"]]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    namespace = kb.ontology()._onto.base_iri
    if kb_name == 'vicodi':
        namespace = 'http://vicodi.org/ontology#'
    print("KB namespace: ", namespace)
    print()
    syntax_checker = SyntaxChecker(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    All_individuals = set(kb.individuals())
    for combine in All_metrics.keys():     
        t0 = time.time()
        inds = [int(model_name.split("I")[-1]) for model_name in combine.split("+")]
        predictions, targets = synthesize_class_expressions(kb_name, inds, args)
        t1 = time.time()
        duration = (t1-t0)/len(predictions)
        for i, pb_str in enumerate(targets):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = 1
            pred = predictions[i][:end_idx]
            #print("Before parsing: ", pred.sum())
            succeed = False
            if (pred=='(').sum() > (pred==')').sum():
                for i in range(len(pred))[::-1]:
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,')')))
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
            elif (pred==')').sum() > (pred=='(').sum():
                for i in range(len(pred)):
                    try:
                        prediction = dl_parser.parse_expression("".join(pred.tolist().insert(i,'(')))
                        succeed = True
                        break
                    except Exception:
                        pass
                if not succeed:
                    try:
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
            else:
                try:
                    prediction = dl_parser.parse_expression("".join(pred.tolist()))
                except Exception:
                    try:
                        pred = syntax_checker.correct(predictions[i].sum())
                        pred = list(syntax_checker.get_suggestions(pred))[-1]
                        prediction = syntax_checker.get_concept(pred)
                    except Exception:
                        prediction = None
            if prediction is None:
                prediction = syntax_checker.get_concept(['⊤'])
            try:
                target_expression = dl_parser.parse_expression(pb_str) # The target class expression
            except:
                continue
            positive_examples = {ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals(target_expression)}
            negative_examples = All_individuals-positive_examples
            try:
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            except Exception as e:
                print("Error on ", prediction)
                print(e)
                prediction = syntax_checker.get_concept(['⊤'])
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            #acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            if verbose:
                print(f'Problem {i}, Target: {pb_str}, Prediction: {syntax_checker.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
                print()
            All_metrics[combine]['acc']['values'].append(acc)
            try:
                All_metrics[combine]['prediction']['values'].append(syntax_checker.renderer.render(prediction))
            except:
                All_metrics[combine]['prediction']['values'].append("Unknown")
            All_metrics[combine]['f1']['values'].append(f1)
            All_metrics[combine]['time']['values'].append(duration)
                                                        
                                                        
    current_iter = 0   
    for lp_type in ['real_values', 'boolean_values', 'inverse_property_values', 'cardinality_restrictions']:
        if not os.path.isfile(f"complex_lps/{kb_name}/{lp_type}.json"): continue
        all_metrics = {'+'.join(combine): defaultdict(lambda: defaultdict(list)) for combine in [["SetTransformer_I32", "SetTransformer_I64"], \
                                        ["SetTransformer_I32", "SetTransformer_I128"], ["SetTransformer_I64", "SetTransformer_I128"],\
                                        ["SetTransformer_I32", "SetTransformer_I64", "SetTransformer_I128"]]}
        with open(f"complex_lps/{kb_name}/{lp_type}.json") as file:
            num_lps = len(json.load(file))
        for key1 in all_metrics:
            for key2 in All_metrics[key1]:
                for key3 in All_metrics[key1][key2]:
                    all_metrics[key1][key2][key3] = All_metrics[key1][key2][key3][current_iter:current_iter+num_lps]
                    if key2 != 'prediction':
                        all_metrics[key1][key2]['mean'] = [np.mean(all_metrics[key1][key2][key3])]
                        all_metrics[key1][key2]['std'] = [np.std(all_metrics[key1][key2][key3])]
            print(f'\n LPs of type {lp_type}')
            print(key1+' Speed: {}s +- {} / lp'.format(round(all_metrics[key1]['time']['mean'][0], 2),\
                                                                   round(all_metrics[key1]['time']['std'][0], 2)))
            print(key1+' Avg Acc: {}% +- {} / lp'.format(round(all_metrics[key1]['acc']['mean'][0], 2),\
                                                                   round(all_metrics[key1]['acc']['std'][0], 2)))
            print(key1+' Avg F1: {}% +- {} / lp'.format(round(all_metrics[key1]['f1']['mean'][0], 2),\
                                                        round(all_metrics[key1]['f1']['std'][0], 2)))
            print()
        current_iter += num_lps
        if save_results:
            with open("datasets/"+kb_name+f"/Results/NCES_{args.kb_emb_model}_Ensemble_{lp_type}.json", "w") as file:
                json.dump(all_metrics, file, indent=3, ensure_ascii=False)

        #for metric in All_metrics[combine]:
        #    if metric != 'prediction':
        #        All_metrics[combine][metric]['mean'] = [np.mean(All_metrics[combine][metric]['values'])]
        #        All_metrics[combine][metric]['std'] = [np.std(All_metrics[combine][metric]['values'])]
#
        #print(combine+' Speed: {}s +- {} / lp'.format(round(All_metrics[combine]['time']['mean'][0], 2),\
        #                                                       round(All_metrics[combine]['time']['std'][0], 2)))
        #print(combine+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics[combine]['acc']['mean'][0], 2),\
        #                                                       round(All_metrics[combine]['acc']['std'][0], 2)))
        #print(combine+' Avg F1: {}% +- {} / lp'.format(round(All_metrics[combine]['f1']['mean'][0], 2),\
        #                                                       round(All_metrics[combine]['f1']['std'][0], 2)))
#
        #print()
#
    #if save_results:
    #    with open(f"datasets/{kb_name}/Results/NCES_{args.kb_emb_model}_Ensemble_{lp_type}.json", "w") as file:
    #        json.dump(All_metrics, file, indent=3, ensure_ascii=False)