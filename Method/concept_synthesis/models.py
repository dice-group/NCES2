import torch, torch.nn as nn, numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce
import os, random

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
        self.loss = nn.CrossEntropyLoss()
        self.encoder = nn.LSTM(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.decoder = nn.LSTM(kwargs['embedding_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.embedding = nn.Embedding(kwargs['output_size']+2, kwargs['embedding_dim'])
        #self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 5*kwargs['rnn_n_hidden']),
        #                        nn.Linear(5*kwargs['rnn_n_hidden'], 2*kwargs['rnn_n_hidden']), 
        #                        nn.Linear(2*kwargs['rnn_n_hidden'],kwargs['output_size']+2))
        self.vocab = ['Start']+kwargs['vocab']+['End']
        self.fc = nn.Linear(kwargs['rnn_n_hidden'], len(self.vocab))
        self.token_to_idx = {tk: i for i,tk in enumerate(self.vocab)}
        
        
    def compute_loss(self, tg, h):
        tg = ['Start'] + tg + ['End']
        tg_ = []
        for i in range(len(tg)-1):
            tg_.append(tg[i])
            inp = self.embedding(torch.tensor(list(map(lambda x: self.token_to_idx[x], tg_))).unsqueeze(0).to(self.device))
            label = torch.tensor(self.token_to_idx[tg[i+1]]).unsqueeze(0).long()
            out, h = self.decoder(inp, h)
            scores = torch.sigmoid(self.fc(out[:, -1, :].view(1, -1)))
            if i == 0:
                loss = self.loss(scores, label)
            else:
                loss += self.loss(scores, label)
        return loss
    
    def forward_compute(self, h):
        aligned_chars = []
        start, end = 'Start', 'End'
        for i in range(h[0].shape[1]):
            preds = [start]
            sampled = start; it = 0
            h_dec = (h[0][:, i, :].unsqueeze(1).contiguous(), h[1][:, i, :].unsqueeze(1).contiguous())
            while sampled != end and it < self.kwargs['max_num_tokens']:
                embs = self.embedding(torch.tensor(list(map(lambda x: self.token_to_idx[x], preds))).to(self.device))
                out, h_dec = self.decoder(embs.reshape(1, embs.shape[0], embs.shape[1]), h_dec)
                scores = torch.sigmoid(self.fc(out[:, -1, :].view(1, -1)))
                id_max = scores.argmax(1)
                sampled = self.vocab[id_max]
                preds.append(sampled)
                it += 1
            if preds[-1] == end:
                preds.pop()
            if len(preds) > 1:
                preds.pop(0)
            aligned_chars.append(np.array(preds, dtype=object).sum())
        return aligned_chars
    
    def forward(self, source, target=None):
        _, h = self.encoder(source)
        #preds = self.forward_compute(h)
        loss = None
        if target is not None: #training mode
            for i,tg in enumerate(target):
                h_ = (h[0][:, i, :].unsqueeze(1).contiguous(), h[1][:, i, :].unsqueeze(1).contiguous())
                if i == 0:
                    loss = self.compute_loss(tg, h_)
                else:
                    loss += self.compute_loss(tg, h_)
        return h, loss/(source.shape[0])
            


        
class ConceptLearner_GRU_As_MT(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.name = 'GRU_As_MT'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = nn.CrossEntropyLoss()
        self.encoder = nn.GRU(kwargs['input_size'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.decoder = nn.GRU(kwargs['embedding_dim'], kwargs['rnn_n_hidden'], kwargs['rnn_n_layers'], dropout=kwargs['drop_prob'], batch_first=True)
        self.embedding = nn.Embedding(kwargs['output_size']+2, kwargs['embedding_dim'])
        self.fc = nn.Sequential(nn.Linear(kwargs['rnn_n_hidden'], 5*kwargs['rnn_n_hidden']),
                                nn.Linear(5*kwargs['rnn_n_hidden'], 2*kwargs['rnn_n_hidden']), 
                                nn.Linear(2*kwargs['rnn_n_hidden'],kwargs['output_size']+2))
        self.vocab = ['Start']+kwargs['vocab']+['End']
        self.token_to_idx = {tk: i for i,tk in enumerate(self.vocab)}
    
    def forward(self, source, target=None):
        _, h = self.encoder(source)
        if target is None: # inference mode
            aligned_chars = []
            start, end = 'Start', 'End'
            for i in range(source.shape[0]):
                preds = [start]
                sampled = start; it = 0
                h_dec = (h[0][:, i, :].unsqueeze(1).contiguous(), h[1][:, i, :].unsqueeze(1).contiguous())
                while sampled != end and it < self.kwargs['max_num_tokens']:
                    embs = self.embedding(torch.tensor(list(map(lambda x: self.token_to_idx[x], preds))).to(self.device))
                    out, _ = self.decoder(embs.reshape(1, embs.shape[0], embs.shape[1]), h_dec)
                    scores = torch.sigmoid(self.fc(out.sum(1).view(1, -1)), 1)
                    id_max = scores.argmax(1)
                    sampled = self.vocab[id_max]
                    preds.append(sampled)
                    it += 1
                if preds[-1] == end:
                    preds.pop()
                if len(preds) > 1:
                    preds.pop(0)
                aligned_chars.append(np.array(preds, dtype=object).sum())
            return aligned_chars
        else: #training mode
            aligned_chars = []
            start, end = 'Start', 'End'
            lengths = [len(tg)+2 for tg in target]
            max_len = max(lengths)-2
            tg_inputs = torch.cat([self.embedding(torch.tensor(list(map(lambda x: self.token_to_idx[x], 
                                    [start]+tg+[end]+[end]*(max_len-len(tg)))))).unsqueeze(0) for tg in target], dim=0).to(self.device)
            tg_inputs = pack_padded_sequence(tg_inputs, lengths, batch_first=True, enforce_sorted=False) #(input, lengths, batch_first=False, enforce_sorted=True)
            out, _ = self.decoder(tg_inputs, h)
            out, _ = pad_packed_sequence(out, batch_first=True)
            scores = torch.sigmoid(self.fc(out.sum(1).view(out.shape[0], -1)), 1)
            self.Loss = self.loss(scores, torch.tensor([self.token_to_idx[end]]*scores.shape[0]).long()).to(self.device)
            return self.Loss

    
        
        