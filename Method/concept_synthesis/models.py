import torch, torch.nn as nn, numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os, random
import pandas as pd

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
    
    
class ConceptLearner_LSTM(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM'
        
        self.lstm = nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x, target_scores=None):
        r_out, _ = self.lstm(x)
        out = r_out.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>0.8*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out

        
class ConceptLearner_GRU(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU'
        
        self.gru = nn.GRU(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())
    
    def forward(self, x, target_scores=None):
        r_out, _ = self.gru(x)
        out = r_out.sum(1).view(-1, self.kwargs['rnn_n_hidden'])
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>0.8*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out

class ConceptLearner_CNN(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'CNN'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(kwargs['kernel_h'],kwargs['kernel_w']), stride=(kwargs['stride_h'],kwargs['stride_w']))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(kwargs['kernel_h']+1,kwargs['kernel_w']), stride=(kwargs['stride_h']+2,kwargs['stride_w']+1))
        self.dropout2d = nn.Dropout2d(kwargs['drop_prob'])
        self.fc = nn.Sequential(nn.Linear(kwargs['conv_out'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']), nn.ReLU(),
                                nn.Linear(20*kwargs['rnn_n_hidden'], 20*kwargs['rnn_n_hidden']), nn.BatchNorm1d(20*kwargs['rnn_n_hidden']),
                                nn.Linear(20*kwargs['rnn_n_hidden'], kwargs['output_size']*kwargs['max_num_atom_repeat']), nn.ReLU())  
    def forward(self, x, target_scores=None):
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.dropout2d(out)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        #print("shape", x.shape)
        out = self.fc(out).reshape(x.shape[0], len(self.kwargs['vocab']), self.kwargs['max_num_atom_repeat'])
        values, sorted_indices = out.flatten(start_dim=1,end_dim=-1).sort(descending=True)
        aligned_chars = []
        if target_scores is None:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>self.kwargs['index_score_upper_bound']*(1-self.kwargs['index_score_lower_bound_rate'])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        else:
            for i in range(sorted_indices.shape[0]):
                num_select = max(1,(out[i]>0.8*min(target_scores[i][target_scores[i]!=0.])).sum().item())
                atoms = []
                stop = 0
                while stop < num_select:
                    v = values[i][stop]
                    try:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0].item()])
                    except ValueError:
                        atoms.append(self.kwargs['vocab'][(out[i]==v).nonzero().squeeze()[0][0].item()])
                    stop += 1
                aligned_chars.append(np.array(atoms, dtype=object).sum())
        return aligned_chars, out

    

class ConceptLearner_LSTM_As_MT(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'LSTM_As_MT'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.decoder = nn.LSTM(kwargs['embedding_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.embedding = nn.Embedding(kwargs['output_size']+3, kwargs['embedding_dim'])
        self.vocab = ['Pad']+['Start']+kwargs['vocab']+['End']
        self.fc = nn.Linear(kwargs['rnn_n_hidden'], len(self.vocab))
        self.token_to_idx = {tk: i for i,tk in enumerate(self.vocab)}
        self.loss = nn.CrossEntropyLoss(ignore_index=self.token_to_idx['Pad'])
        
        
    def compute_loss(self, hidden, target):
        start, end, pad = 'Start', 'End', 'Pad'
        max_len = max([len(tg) for tg in target])+2
        target = np.array([[start]+tg+[end]+[pad]*(max_len-len(tg)) for tg in target], dtype=object)
        for i in range(target.shape[1]-1):
            tg_ = target[:,:i+1]
            inp = pd.DataFrame(tg_).applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
            inp = torch.from_numpy(np.concatenate(inp.values.tolist()))
            inp = inp.reshape(hidden[0].shape[1], tg_.shape[1], -1).requires_grad_().to(self.device)
            label = list(map(self.token_to_idx.get, target[:, i+1]))
            label = torch.tensor(label).long().to(self.device)
            out, hidden = self.decoder(inp, hidden)
            scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
            if i == 0:
                loss = self.loss(scores, label)
            else:
                loss += self.loss(scores, label)
        return loss/len(target)
        
    
    @staticmethod
    def get_pred(pred):
        pred = pred.tolist()
        pred.pop(0)
        for i in range(len(pred)):
            if pred[i] in ['Pad', 'End']:
                return pred[:i]
        return pred
        
    def forward_compute(self, hidden, target=None):
        start, end, pad = 'Start', 'End', 'Pad'
        max_len = max([len(tg) for tg in target])+2 if target is not None else self.kwargs['max_num_tokens']+2
        for i in range(max_len-1):
            if i == 0:
                preds = np.array([[start]*hidden[0].shape[1]]).reshape(-1,1)
                embs = pd.DataFrame(preds)
                embs = embs.applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
                embs = torch.from_numpy(np.concatenate(embs.values.tolist()))
                embs = embs.reshape(hidden[0].shape[1], preds.shape[1], -1).requires_grad_().to(self.device)
                out, hidden = self.decoder(embs, hidden)
                scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
                id_max = scores.argmax(1).tolist()
                sampled = np.array(self.vocab)[id_max]
                preds = np.concatenate([preds, sampled.reshape(-1,1)], axis=1)
            else:
                embs = pd.DataFrame(preds)
                embs = embs.applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
                embs = torch.from_numpy(np.concatenate(embs.values.tolist()))
                embs = embs.reshape(hidden[0].shape[1], preds.shape[1], -1).requires_grad_().to(self.device)
                out, hidden = self.decoder(embs, hidden)
                scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
                id_max = scores.argmax(1).tolist()
                sampled = np.array(self.vocab)[id_max]
                preds = np.concatenate([preds, sampled.reshape(-1,1)], axis=1)
        return np.array(list(map(self.get_pred, preds)), dtype=object).sum(1)
    
    def forward(self, source, target=None):
        _, hidden = self.encoder(source)
        loss = None
        if target is not None:
            loss = self.compute_loss(hidden, target)
            prediction = self.forward_compute(hidden, target)
        else:
            prediction = self.forward_compute(hidden)
        return loss, prediction
            

class ConceptLearner_GRU_As_MT(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU_As_MT'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.CrossEntropyLoss()
        self.encoder = nn.GRU(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.decoder = nn.GRU(kwargs['embedding_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.embedding = nn.Embedding(kwargs['output_size']+3, kwargs['embedding_dim'])
        self.vocab = ['Pad']+['Start']+kwargs['vocab']+['End']
        self.fc = nn.Linear(kwargs['rnn_n_hidden'], len(self.vocab))
        self.token_to_idx = {tk: i for i,tk in enumerate(self.vocab)}
        self.loss = nn.CrossEntropyLoss(ignore_index=self.token_to_idx['Pad'])
        
        
    def compute_loss(self, hidden, target):
        start, end, pad = 'Start', 'End', 'Pad'
        max_len = max([len(tg) for tg in target])+2
        target = np.array([[start]+tg+[end]+[pad]*(max_len-len(tg)) for tg in target], dtype=object)
        for i in range(target.shape[1]-1):
            tg_ = target[:,:i+1]
            inp = pd.DataFrame(tg_).applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
            inp = torch.from_numpy(np.concatenate(inp.values.tolist()))
            inp = inp.reshape(hidden.shape[1], tg_.shape[1], -1).requires_grad_().to(self.device)
            label = list(map(self.token_to_idx.get, target[:, i+1]))
            label = torch.tensor(label).long().to(self.device)
            out, hidden = self.decoder(inp, hidden)
            scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
            if i == 0:
                loss = self.loss(scores, label)
            else:
                loss += self.loss(scores, label)
        return loss/len(target)
        
    
    @staticmethod
    def get_pred(pred):
        pred = pred.tolist()
        pred.pop(0)
        for i in range(len(pred)):
            if pred[i] in ['Pad', 'End']:
                return pred[:i]
        return pred
        
    def forward_compute(self, hidden, target=None):
        start, end, pad = 'Start', 'End', 'Pad'
        max_len = max([len(tg) for tg in target])+2 if target is not None else self.kwargs['max_num_tokens']+2
        for i in range(max_len-1):
            if i == 0:
                preds = np.array([[start]*hidden.shape[1]]).reshape(-1,1)
                embs = pd.DataFrame(preds)
                embs = embs.applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
                embs = torch.from_numpy(np.concatenate(embs.values.tolist()))
                embs = embs.reshape(hidden.shape[1], preds.shape[1], -1).requires_grad_().to(self.device)
                out, hidden = self.decoder(embs, hidden)
                scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
                id_max = scores.argmax(1).tolist()
                sampled = np.array(self.vocab)[id_max]
                preds = np.concatenate([preds, sampled.reshape(-1,1)], axis=1)
            else:
                embs = pd.DataFrame(preds)
                embs = embs.applymap(lambda x: self.embedding(torch.tensor(self.token_to_idx[x]).to(self.device)).detach().cpu().numpy())
                embs = torch.from_numpy(np.concatenate(embs.values.tolist()))
                embs = embs.reshape(hidden.shape[1], preds.shape[1], -1).requires_grad_().to(self.device)
                out, hidden = self.decoder(embs, hidden)
                scores = torch.sigmoid(self.fc(out[:, -1, :].view(out.shape[0], -1)))
                id_max = scores.argmax(1).tolist()
                sampled = np.array(self.vocab)[id_max]
                preds = np.concatenate([preds, sampled.reshape(-1,1)], axis=1)
        return np.array(list(map(self.get_pred, preds)), dtype=object).sum(1)
    
    def forward(self, source, target=None):
        _, hidden = self.encoder(source)
        loss = None
        if target is not None:
            loss = self.compute_loss(hidden, target)
            prediction = self.forward_compute(hidden, target)
        else:
            prediction = self.forward_compute(hidden)
        return loss, prediction
    
        
        