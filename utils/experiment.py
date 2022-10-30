import numpy as np, copy
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import sys, os, json
base_path = os.path.dirname(os.path.realpath(__file__)).split('utils')[0]
sys.path.append(base_path)
from nces import BaseConceptSynthesis
from .dataloader import *
from .data import Data
from nces.synthesizer import ConceptSynthesizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score
import time
import random
from argparse import Namespace

class Experiment:
    
    def __init__(self, kwargs):
        self.decay_rate = kwargs.decay_rate
        self.clip_value = kwargs.grad_clip_value
        self.num_workers = kwargs.num_workers
        self.batch_size = kwargs.batch_size
        self.kb = kwargs.path_to_triples.split("/")[-3]
        self.kb_embedding_data = Data(kwargs)
        complete_args = vars(kwargs)
        complete_args.update({"num_entities": len(self.kb_embedding_data.entities),\
                              "num_relations": len(self.kb_embedding_data.relations)})
        complete_args = Namespace(**complete_args)
        self.cs = ConceptSynthesizer(complete_args)
        self.kwargs = complete_args
        
    
    def before_pad(self, arg):
        arg_temp = []
        for atm in arg:
            if atm == 'PAD':
                break
            arg_temp.append(atm)
        return arg_temp
    
    
    def compute_accuracy(self, prediction, target):
        def soft(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = set(self.before_pad(BaseConceptSynthesis.decompose(arg1_)))
            else:
                arg1_ = set(self.before_pad(arg1_))
            if isinstance(arg2_, str):
                arg2_ = set(self.before_pad(BaseConceptSynthesis.decompose(arg2_)))
            else:
                arg2_ = set(self.before_pad(arg2_))
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

        def hard(arg1, arg2):
            arg1_ = arg1
            arg2_ = arg2
            if isinstance(arg1_, str):
                arg1_ = self.before_pad(BaseConceptSynthesis.decompose(arg1_))
            else:
                arg1_ = self.before_pad(arg1_)
            if isinstance(arg2_, str):
                arg2_ = self.before_pad(BaseConceptSynthesis.decompose(arg2_))
            else:
                arg2_ = self.before_pad(arg2_)
            return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc
          

    def get_optimizer(self, synthesizer, embedding_model, optimizer='Adam'):
        if optimizer == 'Adam':
            return torch.optim.Adam(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        elif optimizer == 'SGD':
            return torch.optim.SGD(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs.learning_rate)
        else:
            raise ValueError
            print('Unsupported optimizer')
    
    def show_num_learnable_params(self, synthesizer, embedding_model):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in synthesizer.parameters()])
        size_ = sum([p.numel() for p in embedding_model.parameters()])
        print(f"Synthesizer ({synthesizer.name} with {synthesizer.num_inds} inducing points): {size}")
        print(f"Embedding Model ({embedding_model.name} with {synthesizer.embedding_dim} embedding dimensions): {size_}")
        print("*"*20+"Trainable model size"+"*"*20)
        print()
        return size, size_
    
    def get_data_idxs(self, data):
        data_idxs = [(self.kb_embedding_data.entity2idx.loc[t[0]].values[0],
                      self.kb_embedding_data.relation2idx.loc[t[1]].values[0],
                      self.kb_embedding_data.entity2idx.loc[t[2]].values[0]) for t in data]
        return data_idxs
    
    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
        
    @staticmethod
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
            
        
    def map_to_token(self, idx_array):
        return self.cs.model.inv_vocab[idx_array]
    
    def train(self, train_data, test_data, epochs=200, test=False, save_model = False, kb_emb_model="TransE",\
              optimizer = 'Adam', record_runtime=False, final=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*100)
        print()
        print("{} starts training on {} data set \n".format(self.cs.model.name, self.kwargs.knowledge_base_path.split("/")[-2]))
        print("#"*100, "\n")
        
        ## Make a copy of the model (initialization)
        synthesizer = copy.deepcopy(self.cs.model)
        embedding_model = copy.deepcopy(self.cs.embedding_model)
        
        ## Initialize data loader for embedding compuation
        triple_data_idxs = self.get_data_idxs(self.kb_embedding_data.data_triples)
        head_to_relation_batch = list(DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.kb_embedding_data.entities)),
            batch_size=2*self.batch_size, num_workers=self.num_workers, shuffle=True))
        
        ## Get combined model size
        size1, size2 = self.show_num_learnable_params(synthesizer, embedding_model)
        desc1 = kb_emb_model+'_'+synthesizer.name
        desc2 = synthesizer.name+'_'+ kb_emb_model+'_'+'Emb'
        if final:
            desc1 = desc1+'_final'
            desc2 = desc2+'_final'
        if train_on_gpu:
            synthesizer.cuda()
            embedding_model.cuda()
                        
        opt = self.get_optimizer(synthesizer=synthesizer, embedding_model=embedding_model, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        Train_acc = defaultdict(list)
        best_score = 0.
        
        if record_runtime:
            t0 = time.time()
            
        train_dataset = CSDataLoader(train_data, self.kwargs)
        tc_batch_iterator = 0
        for e in range(epochs):
            train_dataset.load_embeddings(embedding_model)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=True)
            soft_acc, hard_acc = [], []
            train_losses = []
            for x1, x2, labels in tqdm(train_dataloader):
                ## Compute KG embedding loss
                head_batch = head_to_relation_batch[tc_batch_iterator%len(head_to_relation_batch)]
                e1_idx, r_idx, tc_targets = head_batch
                if train_on_gpu:
                    tc_targets = tc_targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()
                if tc_batch_iterator and tc_batch_iterator%len(head_to_relation_batch) == 0:
                    random.shuffle(head_to_relation_batch)
                tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                tc_batch_iterator += 1
                
                target_sequence = self.map_to_token(labels)
                if train_on_gpu:
                    x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
                pred_sequence, scores = synthesizer(x1, x2)
                cs_loss = synthesizer.loss(scores, labels)
                loss = 0.5 * (tc_loss + cs_loss)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                clip_grad_value_(embedding_model.parameters(), clip_value=self.clip_value)
                opt.step()
                if self.decay_rate:
                    self.scheduler.step()
            tr_soft_acc, tr_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            Train_loss.append(np.mean(train_losses))
            Train_acc['soft'].append(tr_soft_acc)
            Train_acc['hard'].append(tr_hard_acc)
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Train loss: {:.4f}...".format(np.mean(train_losses)),
                  "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                  "Train hard acc: {:.2f}%...".format(tr_hard_acc))
            if np.random.rand() > 0.7:
                print("Visualizing some prediction: ", pred_sequence[np.random.choice(range(x1.shape[0]))])
                print()
            weights_cs = copy.deepcopy(synthesizer.state_dict())
            weights_tc = copy.deepcopy(embedding_model.state_dict())
            if Train_acc['hard'] and Train_acc['hard'][-1] > best_score:
                best_score = Train_acc['hard'][-1]
                best_weights_cs = weights_cs
                best_weights_tc = weights_tc
        synthesizer.load_state_dict(best_weights_cs)
        embedding_model.load_state_dict(best_weights_tc)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Concept synthesizer": synthesizer.name,
                           "Number of Epochs": epochs, "Runtime (s)": duration}
            if not os.path.exists(base_path+f"datasets/{self.kb}/Runtime/"):
                os.mkdir(base_path+f"datasets/{self.kb}/Runtime")
            with open(base_path+f"datasets/{self.kb}/Runtime/"+"Runtime_"+desc1+f"_inducing_points{synthesizer.num_inds}.json", "w") as file:
                json.dump(runtime_info, file, indent=3)
                
        results_dict = dict()
        if test:
            test_dataset = CSDataLoader(test_data, self.kwargs)
            test_dataset.load_embeddings(embedding_model)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_batch, shuffle=False)
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            soft_acc, hard_acc = [], []
            for x1, x2, labels in test_dataloader:
                if train_on_gpu:
                    x1, x2 = x1.cuda(), x2.cuda()
                pred_sequence, _ = synthesizer(x1, x2)
                target_sequence = target_sequence = self.map_to_token(labels)
                s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Synthesizer size": size1, "Embedding model size": size2, "Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        print("Train soft accuracy: {} ... Train hard accuracy: {}".format(max(Train_acc['soft']), max(Train_acc['hard'])))
        print()
        results_dict.update({"Train max soft acc": max(Train_acc['soft']), "Train max hard acc": max(Train_acc['hard']), "Train min loss": min(Train_loss)})
        results_dict.update({'Vocab size': len(synthesizer.vocab)})
        if not os.path.exists(base_path+f"datasets/{self.kb}/Results/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Results/")
        with open(base_path+f"datasets/{self.kb}/Results/"+"Train_Results_"+desc1+f"_inducing_points{synthesizer.num_inds}.json", "w") as file:
                json.dump(results_dict, file, indent=3)
        self.kb_embedding_data.entity2idx.to_csv(base_path+f"datasets/{self.kb}/Model_weights/"+desc2+\
                                                 f"_inducing_points{synthesizer.num_inds}_entity_idx.csv")
        self.kb_embedding_data.relation2idx.to_csv(base_path+f"datasets/{self.kb}/Model_weights/"+desc2+\
                                                   f"_inducing_points{synthesizer.num_inds}_relation_idx.csv")
        if save_model:
            if not os.path.exists(base_path+f"datasets/{self.kb}/Model_weights/"):
                os.mkdir(base_path+f"datasets/{self.kb}/Model_weights/")
            torch.save(synthesizer, base_path+f"datasets/{self.kb}/Model_weights/"+desc1+f"_inducing_points{synthesizer.num_inds}.pt")
            torch.save(embedding_model, base_path+f"datasets/{self.kb}/Model_weights/"+desc2+f"_inducing_points{synthesizer.num_inds}.pt")
            print("{} and {} saved".format(synthesizer.name, embedding_model.name))
            print()
        plot_data = (np.array(Train_acc['soft']), np.array(Train_acc['hard']), Train_loss)
        return plot_data
            
            
    def train_all_nets(self, List_nets, train_data, test_data, epochs=200, test=False, save_model = False, kb_emb_model='TransE', optimizer = 'Adam', record_runtime=False, final=False):
        if not os.path.exists(base_path+f"datasets/{self.kb}/Training_curves/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Training_curves/")
        if not os.path.exists(base_path+f"datasets/{self.kb}/Plot_data/"):
            os.mkdir(base_path+f"datasets/{self.kb}/Plot_data/")
                        
        for net in List_nets:
            self.cs.learner_name = net
            desc = kb_emb_model+'_'+net
            self.cs.refresh()
            train_soft_acc, train_hard_acc, train_l = self.train(train_data, test_data, epochs, test, save_model, kb_emb_model, optimizer, record_runtime, final)
            with open(base_path+f"datasets/{self.kb}/Plot_data/"+desc+f"_inducing_points{self.cs.model.num_inds}.json", "w") as plot_file:
                json.dump({"soft acc": list(train_soft_acc), "hard acc": list(train_hard_acc), "loss": list(train_l)}, plot_file, indent=3)

            
