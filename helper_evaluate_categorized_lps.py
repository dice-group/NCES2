import os, random
from utils.syntax_checker import SyntaxChecker
from utils.evaluator import Evaluator
from ontolearn.knowledge_base import KnowledgeBase
from nces import BaseConceptSynthesis
from nces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from dataloader import CSDataLoader
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

def map_to_token(model, idx_array):
    return model.inv_vocab[idx_array]

def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    target_tokens_list = []
    target_labels = []
    for pos_emb, neg_emb, label in batch:
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
        target_labels.append(label)
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
    target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
    return pos_emb_list, neg_emb_list, target_labels

def predict(kb, models, args, lp_type):
    data_test_path = f"datasets/{kb}/Test_data/{lp_type}.json"
    with open(data_test_path, "r") as file:
        test_data = json.load(file)
    print("\n*** Number of learning problems: ", len(test_data), "***\n")
    args.path_to_triples = f"datasets/{kb}/Triples/" ## Loads the smallest dataset triples by default. Only required to intitalize CSDataLoader
    test_dataset = CSDataLoader(test_data, args)
    soft_acc, hard_acc = 0.0, 0.0
    preds = []
    targets = []
    if isinstance(models, list):
        for i, model in enumerate(models):
            model.eval()
            scores = []
            num_inds = model.num_inds
            embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                        map_location = torch.device("cpu"))
            test_dataset.load_embeddings(embedding_model)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
            for x1, x2, labels in tqdm(test_dataloader):
                if i == 0:
                    target_sequence = map_to_token(model, labels)
                    targets.append(target_sequence)
                _, sc = model(x1, x2)
                scores.append(sc) 
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
        models.eval()
        num_inds = models.num_inds
        embedding_model = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{num_inds}.pt",
                                    map_location = torch.device("cpu"))
        test_dataset.load_embeddings(embedding_model)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
        for x1, x2, labels in tqdm(test_dataloader):
            target_sequence = map_to_token(models, labels)
            pred_sequence, _ = models(x1, x2)
            preds.append(pred_sequence)
            targets.append(target_sequence)
            s_acc, h_acc = compute_accuracy(pred_sequence, target_sequence)
            soft_acc += s_acc
            hard_acc += h_acc
        print(f"Average syntactic accuracy, Soft: {soft_acc/len(test_dataloader)}%, Hard: {hard_acc/len(test_dataloader)}%")
        return np.concatenate(preds, 0), np.concatenate(targets, 0)

def synthesize_class_expressions(kb, num_inds, args, lp_type):
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    if isinstance(num_inds, list):
        print(f"\n## Ensemble prediction ({'+'.join([f'SetTransformer_I{inds}' for inds in num_inds])})")
    
        models = [torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{inds}.pt",
                             map_location=torch.device("cpu")) for inds in num_inds]
        return predict(kb, models, args, lp_type)
    else:
        print(f"\n## Single model prediction (SetTransformer_I{num_inds})")
        model = torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{num_inds}.pt",
                           map_location=torch.device("cpu"))
        return predict(kb, model, args, lp_type)
    
    

def evaluate_nces(kb_name, num_inds, args, lp_type, save_results=False, verbose=False):
    print('#'*50)
    print('NCES evaluation on {} LPs of {} KB:'.format(lp_type, kb_name))
    print('#'*50)
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
    if not os.path.isfile(f"datasets/{kb_name}/Test_data/{lp_type}.json"): return
    for model_name in All_metrics.keys():
        num_inds = int(model_name.split("I")[-1])
        t0 = time.time()
        predictions, targets = synthesize_class_expressions(kb_name, num_inds, args, lp_type)
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
                        print(f"Could not understand expression {pred}")
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
                        print(f"Could not understand expression {pred}")
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
                        print(f"Could not understand expression {pred}")
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
            
        for metric in All_metrics[model_name]:
            if metric != 'prediction':
                All_metrics[model_name][metric]['mean'] = [np.mean(All_metrics[model_name][metric]['values'])]
                All_metrics[model_name][metric]['std'] = [np.std(All_metrics[model_name][metric]['values'])]
        
        print(model_name+' Speed: {}s +- {} / lp'.format(round(All_metrics[model_name]['time']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['time']['std'][0], 2)))
        print(model_name+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics[model_name]['acc']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['acc']['std'][0], 2)))
        print(model_name+' Avg F1: {}% +- {} / lp'.format(round(All_metrics[model_name]['f1']['mean'][0], 2),\
                                                               round(All_metrics[model_name]['f1']['std'][0], 2)))
        print()
        if save_results:
            with open("datasets/"+kb_name+f"/Results/NCES_{args.kb_emb_model}_{lp_type}"+".json", "w") as file:
                json.dump(All_metrics, file, indent=3, ensure_ascii=False)

                
def evaluate_ensemble(kb_name, args, lp_type, save_results=False, verbose=False):
    print('#'*50)
    print('NCES evaluation on {} LPs of {} KB:'.format(lp_type, kb_name))
    print('#'*50)
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
    if not os.path.isfile(f"datasets/{kb_name}/Test_data/{lp_type}.json"): return
    for combine in All_metrics.keys():     
        t0 = time.time()
        num_inds = [int(model_name.split("I")[-1]) for model_name in combine.split("+")]
        predictions, targets = synthesize_class_expressions(kb_name, num_inds, args, lp_type)
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
                        print(f"Could not understand expression {pred}")
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
                        print(f"Could not understand expression {pred}")
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
                        print(f"Could not understand expression {pred}")
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

        for metric in All_metrics[combine]:
            if metric != 'prediction':
                All_metrics[combine][metric]['mean'] = [np.mean(All_metrics[combine][metric]['values'])]
                All_metrics[combine][metric]['std'] = [np.std(All_metrics[combine][metric]['values'])]

        print(combine+' Speed: {}s +- {} / lp'.format(round(All_metrics[combine]['time']['mean'][0], 2),\
                                                               round(All_metrics[combine]['time']['std'][0], 2)))
        print(combine+' Avg Acc: {}% +- {} / lp'.format(round(All_metrics[combine]['acc']['mean'][0], 2),\
                                                               round(All_metrics[combine]['acc']['std'][0], 2)))
        print(combine+' Avg F1: {}% +- {} / lp'.format(round(All_metrics[combine]['f1']['mean'][0], 2),\
                                                               round(All_metrics[combine]['f1']['std'][0], 2)))

        print()

    if save_results:
        with open(f"datasets/{kb_name}/Results/NCES_{args.kb_emb_model}_Ensemble_{lp_type}.json", "w") as file:
            json.dump(All_metrics, file, indent=3, ensure_ascii=False)
