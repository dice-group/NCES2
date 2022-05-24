import numpy as np, copy
import torch, random
from collections import Counter, defaultdict
from sklearn.utils import resample
from torch.utils.data import DataLoader
import sys, os, json
base_path = os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0]
sys.path.append(base_path)
# from util.weightedloss import WeightedMSELoss
from helper_classes.dataloader import HeadAndRelationBatchLoader, CSDataLoader
from concept_synthesis.helper_classes import ConceptSynthesizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_value_
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import f1_score, accuracy_score
import time

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Experiment:
    
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.decay_rate = kwargs['decay_rate']
        self.clip_value = kwargs['grad_clip_value']
        self.cs = ConceptSynthesizer(kwargs)
        self.loss = MSELoss()
    
    def get_data_idxs(self, data):
        data_idxs = [(self.cs.dataloader.entity_to_idx[t[0]], self.cs.dataloader.relation_to_idx[t[1]], self.cs.dataloader.entity_to_idx[t[2]]) for t in data]
        return data_idxs
    
    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    
    def compute_accuracy(self, prediction:str, target:str):
        def soft(arg1, arg2):
            arg1_ = set(self.cs.dataloader.decompose(arg1)) - {' '}
            arg2_ = set(self.cs.dataloader.decompose(arg2)) - {' '}
            return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))
        
        def hard(arg1, arg2):
            arg1_ = self.cs.dataloader.decompose(arg1)
            arg2_ = self.cs.dataloader.decompose(arg2)
            return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
        soft_acc = sum(map(soft, prediction, target))/len(target)
        hard_acc = sum(map(hard, prediction, target))/len(target)
        return soft_acc, hard_acc
          
        
    def get_batch(self, x, y, z, batch_size, shuffle=True):
        if shuffle:
            indx = list(range(x.shape[0]))
            random.shuffle(indx)
            x, y, z = x[indx], y[indx], z[indx]
        if len(x) >= batch_size:
            for i in range(0, x.shape[0]-batch_size+1, batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size], z[i:i+batch_size]
        else:
            yield x, y, z
            
    def get_optimizer(self, synthesizer, optimizer='Adam', embedding_model=None):
        if embedding_model is not None:
            if optimizer == 'Adam':
                return torch.optim.Adam(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            elif optimizer == 'SGD':
                return torch.optim.SGD(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            elif optimizer == 'RMSprop':
                return torch.optim.RMSprop(list(synthesizer.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            else:
                raise ValueError
                print('Unsupported optimizer')
        else:
            if optimizer == 'Adam':
                return torch.optim.Adam(synthesizer.parameters(), lr=self.kwargs['learning_rate'])
            elif optimizer == 'SGD':
                return torch.optim.SGD(synthesizer.parameters(), lr=self.kwargs['learning_rate'])
            elif optimizer == 'RMSprop':
                return torch.optim.RMSprop(synthesizer.parameters(), lr=self.kwargs['learning_rate'])
            else:
                raise ValueError
                print('Unsupported optimizer')
    
    def show_num_learnable_params(self):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in self.cs.synthesizer.parameters()])
        size_ = 0
        print("synthesizer: ", size)
        if self.cs.embedding_model is not None:
            size_ += sum([p.numel() for p in self.cs.embedding_model.parameters()])
            size += size_
        print("Embedding model: ", size_)
        print("Total: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
    
    def train(self, data_train, data_test, epochs=200, cs_batch_size=64, tc_batch_size=512, kf_n_splits=10, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False, final=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.show_num_learnable_params()
        if self.cs.embedding_model is not None and include_embedding_loss:
            triple_data_idxs = self.get_data_idxs(self.cs.dataloader.data)
            head_to_relation_batch = list(DataLoader(
                HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.cs.dataloader.entities)),
                batch_size=tc_batch_size, num_workers=12, shuffle=True))
        
        embeddings = None
        if self.cs.embedding_model is None:
            embeddings = self.cs.get_embedding(embedding_model=None)   
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.cs.synthesizer.name, self.kwargs['path_to_triples'].split("/")[-3]))
        print("#"*50, "\n")
        synthesizer = copy.deepcopy(self.cs.synthesizer)
        if self.cs.embedding_model is not None:
                embedding_model = copy.deepcopy(self.cs.embedding_model)
        desc = synthesizer.name
        if final:
            desc = desc+'_final'
        if train_on_gpu:
            synthesizer.cuda()
            if embeddings is None:
                embedding_model.cuda()
        if embeddings is None: 
            opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer, embedding_model=embedding_model)
        else:
            opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        Train_loss = []
        Train_acc = defaultdict(list)
        best_score = 0.
        if include_embedding_loss:
            tc_iterator = 0
        if record_runtime:
            t0 = time.time()
        Emb = embeddings if embeddings is not None else self.cs.get_embedding(embedding_model)
        if self.cs.embedding_model is None:
            for e in range(epochs):
                soft_acc, hard_acc = [], []
                train_losses = []
                for x, y_numerical, target_sequence in self.get_batch(data_train[0], data_train[1], data_train[2], batch_size=cs_batch_size):
                    if(train_on_gpu):
                        x, y_numerical = x.cuda(), y_numerical.cuda()
                    #ipdb.set_trace()
                    if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                        pred_sequence, scores = synthesizer(x, y_numerical)
                        cs_loss = self.loss(scores, y_numerical)
                    else:
                        target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                        h, cs_loss = synthesizer(x, target) # h is the hidden state in LSTM
                        pred_sequence = synthesizer.forward_compute(h)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
                    train_losses.append(cs_loss.item())
                    opt.zero_grad()
                    cs_loss.backward()
                    clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
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
                    print("Visualizing some prediction: ", np.random.choice(pred_sequence))
                weights_cs = copy.deepcopy(synthesizer.state_dict())
                if Train_acc['soft'] and Train_acc['soft'][-1] > best_score:
                    best_score = Train_acc['soft'][-1]
                    best_weights_cs = weights_cs
        else:
            for e in range(epochs):
                soft_acc, hard_acc = [], []
                train_losses = []
                for x, y_numerical, target_sequence in self.cs.dataloader.load(Emb, data=data_train, batch_size=cs_batch_size, shuffle=True):
                    if include_embedding_loss:
                        head_batch = head_to_relation_batch[tc_iterator%len(head_to_relation_batch)]
                        tc_iterator += 1
                        e1_idx, r_idx, tc_targets = head_batch
                        if train_on_gpu:
                            tc_targets = tc_targets.cuda()
                            r_idx = r_idx.cuda()
                            e1_idx = e1_idx.cuda()
                        if tc_label_smoothing:
                            tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))
                        tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                    if(train_on_gpu):
                        x, y_numerical = x.cuda(), y_numerical.cuda()
                    #ipdb.set_trace()
                    if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                        pred_sequence, scores = synthesizer(x, y_numerical)
                        cs_loss = self.loss(scores, y_numerical)
                    else:
                        target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                        h, cs_loss = synthesizer(x, target)
                        pred_sequence = synthesizer.forward_compute(h)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
                    
                    if include_embedding_loss:
                        tcs_loss = 0.5*cs_loss + 0.5*tc_loss
                    else:
                        tcs_loss = cs_loss
                    train_losses.append(tcs_loss.item())
                    opt.zero_grad()
                    tcs_loss.backward()
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
                    print("Visualizing some prediction: ", np.random.choice(pred_sequence))
                weights_cs = copy.deepcopy(synthesizer.state_dict())
                weights_emb = copy.deepcopy(embedding_model.state_dict())                
                if Train_acc['soft'] and Train_acc['soft'][-1] > best_score:
                    best_score = Train_acc['soft'][-1]
                    best_weights_cs = weights_cs
                    best_weights_emb = weights_emb
        synthesizer.load_state_dict(best_weights_cs)
        if self.cs.embedding_model is not None:
            embedding_model.load_state_dict(best_weights_emb)
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Concept synthesizer": synthesizer.name,
                           "Number of Epochs": epochs, "Runtime (s)": duration}
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime")
            if embeddings is None:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime/"+"Runtime_"+embedding_model.name+'_'+desc+".json", "w") as file:
                    json.dump(runtime_info, file, indent=3)
            else:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime/"+"Runtime_"+desc+".json", "w") as file:
                    json.dump(runtime_info, file, indent=3)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            if embeddings is None:
                embedding_model.eval()
            soft_acc, hard_acc = [], []
            if self.cs.embedding_model is None:
                for x, _, target_sequence in self.get_batch(data_test[0], data_test[1], data_test[2], batch_size=cs_batch_size, shuffle=False):
                    if train_on_gpu:
                        x = x.cuda()
                    if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                        pred_sequence, _ = synthesizer(x)
                    else:
                        h, _ = synthesizer(x)
                        pred_sequence = synthesizer.forward_compute(h)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
                    
            else:
                for x, _, target_sequence in self.cs.dataloader.load(Emb, data=data_test, batch_size=cs_batch_size, shuffle=False):
                    if train_on_gpu:
                        x = x.cuda()
                    if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                        pred_sequence, _ = synthesizer(x)
                    else:
                        h, _ = synthesizer(x)
                        pred_sequence = synthesizer.forward_compute(h)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        print("Train soft accuracy: {} ... Train hard accuracy: {}".format(max(Train_acc['soft']), max(Train_acc['hard'])))
        print()
        results_dict.update({"Train Max Soft Acc": max(Train_acc['soft']), "Train Max Hard Acc": max(Train_acc['hard']), "Train Min Loss": min(Train_loss)})
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/")
        if embeddings is None:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+embedding_model.name+'_'+desc+".json", "w") as file:
                json.dump(results_dict, file, indent=3)
        else:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+desc+".json", "w") as file:
                json.dump(results_dict, file, indent=3)
        if save_model:
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/")
            torch.save(synthesizer, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+desc+".pt")
            if embeddings is None:
                torch.save(embedding_model, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+embedding_model.name+'_'+desc+".pt")
            print("{} saved".format(synthesizer.name))
            print()
        plot_data = (np.array(Train_acc['soft']), np.array(Train_acc['hard']), Train_loss)
        return plot_data
        
    
    def cross_validate(self, data_train, data_test, epochs=200, cs_batch_size=64, tc_batch_size=512, kf_n_splits=10, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, *kwargs):
        if self.cs.embedding_model is not None and include_embedding_loss:
            triple_data_idxs = self.get_data_idxs(self.cs.dataloader.data)
            head_to_relation_batch = list(DataLoader(
                HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.cs.dataloader.entities)),batch_size=tc_batch_size, num_workers=12, shuffle=True))
        embeddings = None
        if self.cs.embedding_model is None:
            embeddings = self.cs.get_embedding(embedding_model=None)
            
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")   
        best_score = 0.
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.cs.synthesizer.name, self.kwargs['path_to_triples'].split("/")[-3]))
        print("#"*50, "\n")
        from sklearn.model_selection import KFold
        Kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=142)
        fold = 0
        All_losses = defaultdict(list)
        All_acc = defaultdict(list)
        best_val_score = 0.
        iterable = data_train if self.cs.embedding_model is not None else list(range(len(data_train[1])))
        for train_index, valid_index in Kf.split(iterable):
            self.show_num_learnable_params()
            synthesizer = copy.deepcopy(self.cs.synthesizer)
            embedding_model = None
            if self.cs.embedding_model is not None:
                embedding_model = copy.deepcopy(self.cs.embedding_model)
            if train_on_gpu:
                synthesizer.cuda()
                if embeddings is None:
                    embedding_model.cuda()
            if embeddings is None: 
                opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer, embedding_model=embedding_model)
            else:
                opt = self.get_optimizer(synthesizer=synthesizer, optimizer=optimizer)
            if self.decay_rate:
                self.scheduler = ExponentialLR(opt, self.decay_rate)
            if self.cs.embedding_model is None:
                x_train, x_valid = data_train[0][train_index], data_train[0][valid_index]
                y_train, y_valid = data_train[1][train_index], data_train[1][valid_index]
                z_train, z_valid = data_train[2][train_index], data_train[2][valid_index]
            else:
                d_train, d_valid = np.array(data_train,dtype=object)[train_index], np.array(data_train,dtype=object)[valid_index]
            fold += 1
            print("*"*50)
            print("Fold {}/{}:\n".format(fold, kf_n_splits))
            print("*"*50, "\n")
            Train_losses = []
            Val_losses = []
            Train_acc = defaultdict(list)
            Val_acc = defaultdict(list)
            if self.cs.embedding_model is not None and include_embedding_loss:
                tc_iterator = 0
            Emb = embeddings if embeddings is not None else self.cs.get_embedding(embedding_model)
            if self.cs.embedding_model is None:
                for e in range(epochs):
                    soft_acc, hard_acc = [], []
                    train_losses = []
                    for x, y_numerical, target_sequence in self.get_batch(x_train, y_train, z_train, batch_size=cs_batch_size):
                        if(train_on_gpu):
                            x, y_numerical = x.cuda(), y_numerical.cuda()
                        if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                            pred_sequence, scores = synthesizer(x, y_numerical)
                            cs_loss = self.loss(scores, y_numerical)
                        else:
                            target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                            h, cs_loss = synthesizer(x, target)
                            pred_sequence = synthesizer.forward_compute(h)
                        s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                        soft_acc.append(s_acc); hard_acc.append(h_acc)
                        train_losses.append(cs_loss.item())
                        opt.zero_grad()
                        cs_loss.backward()
                        clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                        opt.step()
                        if self.decay_rate:
                            self.scheduler.step()
                    tr_soft_acc, tr_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                    # Get validation loss
                    val_losses = []
                    synthesizer.eval()
                    soft_acc, hard_acc = [], []
                    for x, y_numerical, target_sequence in self.get_batch(x_valid, y_valid, z_valid, batch_size=cs_batch_size):
                        if(train_on_gpu):
                            x, y_numerical = x.cuda(), y_numerical.cuda()
                        if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                            pred_sequence, scores = synthesizer(x)
                            val_loss = self.loss(scores, y_numerical)
                        else:
                            target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                            h, val_loss = synthesizer(x, target)
                            pred_sequence = synthesizer.forward_compute(h)
                        s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                        soft_acc.append(s_acc); hard_acc.append(h_acc)
                        val_losses.append(val_loss.item())
                    val_soft_acc, val_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                    synthesizer.train() # reset to train mode after iteration through validation data
                    Train_losses.append(np.mean(train_losses))
                    Val_losses.append(np.mean(val_losses))
                    Val_acc['soft'].append(val_soft_acc)
                    Val_acc['hard'].append(val_hard_acc)
                    Train_acc['soft'].append(tr_soft_acc)
                    Train_acc['hard'].append(tr_hard_acc)
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train loss: {:.4f}...".format(np.mean(train_losses)),
                          "Val loss: {:.4f}...".format(np.mean(val_losses)),
                          "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                          "Train hard acc: {:.2f}%...".format(tr_hard_acc),
                          "Val soft acc: {:.2f}%...".format(val_soft_acc),
                          "Val hard acc: {:.2f}%".format(val_hard_acc))
                    weights_cs = copy.deepcopy(synthesizer.state_dict())
                    if Val_acc['soft'] and max(Val_acc['soft']) > best_val_score:
                        best_val_score = max(Val_acc['soft'])
                        best_weights_cs = weights_cs
                    All_losses["train"].append(Train_losses)
                    All_losses["val"].append(Val_losses)
                    All_acc["train"].append(Train_acc)
                    All_acc["val"].append(Val_acc)
            else:
                for e in range(epochs):
                    soft_acc, hard_acc = [], []
                    train_losses = []
                    for x, y_numerical, target_sequence in self.cs.dataloader.load(Emb, data=d_train, batch_size=cs_batch_size, shuffle=True):
                        if include_embedding_loss:
                            head_batch = head_to_relation_batch[tc_iterator%len(head_to_relation_batch)]
                            tc_iterator += 1
                            e1_idx, r_idx, tc_targets = head_batch
                            if train_on_gpu:
                                tc_targets = tc_targets.cuda()
                                r_idx = r_idx.cuda()
                                e1_idx = e1_idx.cuda()
                            if tc_label_smoothing:
                                tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))
                            tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                        if(train_on_gpu):
                            x, y_numerical = x.cuda(), y_numerical.cuda()
                        if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                            pred_sequence, scores = synthesizer(x, y_numerical)
                            cs_loss = self.loss(scores, y_numerical)
                        else:
                            target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                            h, cs_loss = synthesizer(x, target)
                            pred_sequence = synthesizer.forward_compute(h)
                        s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                        soft_acc.append(s_acc); hard_acc.append(h_acc)
                        if include_embedding_loss:
                            tcs_loss = 0.5*cs_loss + 0.5*tc_loss
                        else:
                            tcs_loss = cs_loss
                        train_losses.append(tcs_loss.item())
                        opt.zero_grad()
                        tcs_loss.backward()
                        clip_grad_value_(synthesizer.parameters(), clip_value=self.clip_value)
                        clip_grad_value_(embedding_model.parameters(), clip_value=self.clip_value)
                        opt.step()
                        if self.decay_rate:
                            self.scheduler.step()
                        Emb = embeddings if embeddings is not None else self.cs.get_embedding(embedding_model)
                    tr_soft_acc, tr_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                    # Get validation loss
                    val_losses = []
                    synthesizer.eval()
                    embedding_model.eval()
                    soft_acc, hard_acc = [], []
                    for x, y_numerical, target_sequence in self.cs.dataloader.load(Emb, data=d_valid, batch_size=cs_batch_size, shuffle=False):
                        if(train_on_gpu):
                            x, y_numerical = x.cuda(), y_numerical.cuda()
                        if include_embedding_loss:
                            head_batch = random.choice(head_to_relation_batch)
                            e1_idx, r_idx, tc_targets = head_batch
                            if train_on_gpu:
                                tc_targets = tc_targets.cuda()
                                r_idx = r_idx.cuda()
                                e1_idx = e1_idx.cuda()
                            if tc_label_smoothing:
                                tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))
                            tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                        if not synthesizer.name in ['LSTM_As_MT', 'GRU_As_MT']:
                            pred_sequence, scores = synthesizer(x)
                            val_loss = self.loss(scores, y_numerical)
                        else:
                            target = list(map(lambda x: self.cs.dataloader.decompose(x), target_sequence))
                            h, val_loss = synthesizer(x, target)
                            pred_sequence = synthesizer.forward_compute(h)
                        s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                        soft_acc.append(s_acc); hard_acc.append(h_acc)
                        val_losses.append(val_loss.item())
                    val_soft_acc, val_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
                    synthesizer.train(); embedding_model.train() # reset to train mode after iterationg through validation data
                    Train_losses.append(np.mean(train_losses))
                    Val_losses.append(np.mean(val_losses))
                    Val_acc['soft'].append(val_soft_acc)
                    Val_acc['hard'].append(val_hard_acc)
                    Train_acc['soft'].append(tr_soft_acc)
                    Train_acc['hard'].append(tr_hard_acc)
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train loss: {:.4f}...".format(np.mean(train_losses)),
                          "Val loss: {:.4f}...".format(np.mean(val_losses)),
                          "Train soft acc: {:.2f}%...".format(tr_soft_acc),
                          "Train hard acc: {:.2f}%...".format(tr_hard_acc),
                          "Val soft acc: {:.2f}%".format(val_soft_acc),
                          "Val hard acc: {:.2f}%".format(val_hard_acc))
                weights_cs = copy.deepcopy(synthesizer.state_dict())
                weights_emb = copy.deepcopy(embedding_model.state_dict())
                if Val_acc['soft'] and max(Val_acc['soft']) > best_val_score:
                    best_val_score = max(Val_acc['soft'])
                    best_weights_cs = weights_cs
                    best_weights_emb = weights_emb
                All_losses["train"].append(Train_losses)
                All_losses["val"].append(Val_losses)
                All_acc["train"].append(Train_acc)
                All_acc["val"].append(Val_acc)
        synthesizer.load_state_dict(best_weights_cs)
        if embeddings is None:
            embedding_model.load_state_dict(best_weights_emb)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            synthesizer.eval()
            if embeddings is None:
                embedding_model.eval()
            soft_acc, hard_acc = [], []
            if self.cs.embedding_model is None:
                for x, _, target_sequence in self.get_batch(data_test[0], data_test[1], data_test[2], batch_size=cs_batch_size, shuffle=False):
                    if train_on_gpu:
                        x = x.cuda()
                    pred_sequence, _ = synthesizer(x)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
            else:
                for x, _, target_sequence in self.cs.dataloader.load(Emb, data=data_test, batch_size=cs_batch_size, shuffle=False):
                    if train_on_gpu:
                        x = x.cuda()
                    pred_sequence, _ = synthesizer(x)
                    s_acc, h_acc = self.compute_accuracy(pred_sequence, target_sequence)
                    soft_acc.append(s_acc); hard_acc.append(h_acc)
            te_soft_acc, te_hard_acc = np.mean(soft_acc), np.mean(hard_acc)
            print("Test for {}:".format(synthesizer.name))
            print("Test soft accuracy: ", te_soft_acc)
            print("Test hard accuracy: ", te_hard_acc)
            results_dict.update({"Test soft acc":te_soft_acc, "Test hard acc": te_hard_acc})
        plot_data = (np.array([a['soft'] for a in All_acc['train']]).mean(1), np.array([a['hard'] for a in All_acc['train']]).mean(1),
                     np.array([a['soft'] for a in All_acc['val']]).mean(1), np.array([a['hard'] for a in All_acc['val']]).mean(1),
                     np.array([l for l in All_losses['train']]).mean(1), np.array([l for l in All_losses['val']]).mean(1))
        print("Train soft accuracy: {} ... Train hard accuracy: {} ... Val soft accuracy: {} ... Val hard accuracy: {}".format(max(plot_data[0]), max(plot_data[1]), max(plot_data[2]), max(plot_data[3])))
        print()
        results_dict.update({"Train Max Avg Soft Acc": max(plot_data[0]),
                              "Train Max Avg Hard Acc": max(plot_data[1]),
                              "Val Max Avg Soft Acc": max(plot_data[2]),
                              "Val Max Avg Hard Acc": max(plot_data[3]),
                              "Train Min Avg Loss": min(plot_data[4]),
                              "Val Min Avg Loss": min(plot_data[5])})
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/")
        if embeddings is None:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+embedding_model.name+'_'+synthesizer.name+".json", "w") as file:
                json.dump(results_dict, file, indent=3)
        else:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+synthesizer.name+".json", "w") as file:
                json.dump(results_dict, file, indent=3)

        if save_model:
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/")
            torch.save(synthesizer, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+synthesizer.name+".pt")
            if embeddings is None:
                torch.save(embedding_model, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+embedding_model.name+'_'+synthesizer.name+".pt")
            print("{} saved".format(synthesizer.name))
        return plot_data
    
    
    def train_and_eval(self, data_train, data_test, epochs=200, cs_batch_size=64, tc_batch_size=512, kf_n_splits=10, cross_validate=False, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False, final=False):
        
        """
        function for training a concept length learner in DL KGs
        
        key args
        -> cll_batch_size: batch_size for the concept learner training (cs: concept synthesizer)
        -> tc_batch_size: batch_size for the training of the embedding model (tc: triple classification)
        key args
        """
        if cross_validate:
            return self.cross_validate(data_train, data_test, epochs, cs_batch_size, tc_batch_size,
                                       kf_n_splits, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime, final)

        else:
            return self.train(data_train, data_test, epochs, cs_batch_size, tc_batch_size,
                    kf_n_splits, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime, final)
            
            
    def train_all_nets(self, List_nets, data_train, data_test, epochs=200, cs_batch_size=64, tc_batch_size=512, kf_n_splits=10, cross_validate=False, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False, final=False):
        if self.cs.embedding_model is None:
            embeddings = self.cs.get_embedding(embedding_model=None)
            print("Loading train, validate, and test data\n")
            data_train = self.cs.dataloader.load(embeddings, data=data_train, shuffle=True)
            data_test = self.cs.dataloader.load(embeddings, data=data_test, shuffle=False)
            print("Done loading train, validate, and test data\n")
        Training_data = dict()
        Validation_data = dict()
        Markers = ['--', ':', '-', '.']
        Colors = ['g', 'b', 'm', 'c']
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Training_curves"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Training_curves")
        if cross_validate:
            for net in List_nets:
                self.cs.learner_name = net
                self.cs.refresh()
                train_soft_acc, train_hard_acc, val_soft_acc, val_hard_acc, train_l, val_l = self.train_and_eval(data_train, data_test, epochs, cs_batch_size, tc_batch_size, kf_n_splits, cross_validate, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime, final)
                Training_data.setdefault("soft acc", []).append(list(train_soft_acc))
                Training_data.setdefault("hard acc", []).append(list(train_hard_acc))
                Training_data.setdefault("loss", []).append(list(train_l))
                Validation_data.setdefault("soft acc", []).append(list(val_soft_acc))
                Validation_data.setdefault("hard acc", []).append(list(val_hard_acc))
                Validation_data.setdefault("loss", []).append(list(val_l))

            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/")
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_with_val.json", "w") as plot_file:
                json.dump({'train': Training_data, 'val': Validation_data}, plot_file, indent=3)
        else:
            for net in List_nets:
                self.cs.learner_name = net
                self.cs.refresh()
                train_soft_acc, train_hard_acc, train_l = self.train_and_eval(data_train, data_test, epochs, cs_batch_size, tc_batch_size, kf_n_splits, cross_validate, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime, final)
                Training_data.setdefault("soft acc", []).append(list(train_soft_acc))
                Training_data.setdefault("hard acc", []).append(list(train_hard_acc))
                Training_data.setdefault("loss", []).append(train_l)

            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/")
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data.json", "w") as plot_file:
                json.dump(Training_data, plot_file, indent=3)
