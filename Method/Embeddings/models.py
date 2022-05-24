import torch
from torch.nn import functional as F, Parameter
import numpy as np
from torch.nn.init import xavier_normal_
import torch.nn as nn
import torch.optim as optim
from numpy.random import RandomState

torch.backends.cudnn.deterministic = True
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


# Complex implementation obtained from https://github.com/TimDettmers/ConvE/blob/master/model.py.

class Complex(torch.nn.Module):
    def __init__(self, param):
        super(Complex, self).__init__()
        self.name = 'Complex'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']

        self.Er = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.Rr = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.Ei = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.Ri = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)

        self.input_dropout = torch.nn.Dropout(self.param['input_dropout'])
        self.bn0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.Er.weight.data)
        xavier_normal_(self.Rr.weight.data)
        xavier_normal_(self.Ei.weight.data)
        xavier_normal_(self.Ri.weight.data)

    def forward_head_batch(self, e1_idx, rel_idx):
        e1r = self.Er(e1_idx)
        rr = self.Rr(rel_idx)
        e1i = self.Ei(e1_idx)
        ri = self.Ri(rel_idx)
        e1r = self.bn0(e1r)
        e1r = self.input_dropout(e1r)
        e1i = self.bn1(e1i)
        e1i = self.input_dropout(e1i)
        pred = torch.mm(e1r * rr, self.Er.weight.transpose(1, 0)) + \
               torch.mm(e1r * ri, self.Ei.weight.transpose(1, 0)) + \
               torch.mm(e1i * rr, self.Ei.weight.transpose(1, 0)) - \
               torch.mm(e1i * ri, self.Er.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def get_embeddings(self):
        entity_emb = torch.cat((self.Er.weight.data, self.Ei.weight.data), 1)
        rel_emb = torch.cat((self.Rr.weight.data, self.Ri.weight.data), 1)
        return entity_emb, rel_emb


class ConEx(torch.nn.Module):
    """ Convolutional Complex Knowledge Graph Embeddings"""

    def __init__(self, params=None):
        super(ConEx, self).__init__()
        self.name = 'ConEx'
        self.loss = torch.nn.BCELoss()
        self.param = params
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']
        self.kernel_size = params['kernel_size']
        self.num_of_output_channels = params['num_of_output_channels']

        # Embeddings.
        self.emb_ent_real = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # imaginary i

        self.emb_rel_real = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # imaginary i

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_ent_i = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_i = torch.nn.Dropout(self.param['input_dropout'])
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        conv_out_dim = ((4+2-self.kernel_size)+1)*((self.embedding_dim+2-self.kernel_size)+1)
        self.fc_num_input = conv_out_dim * self.num_of_output_channels
        self.fc = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 2)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(self.embedding_dim * 2)
        self.feature_map_dropout = torch.nn.Dropout2d(self.param['feature_map_dropout'])

    def residual_convolution(self, C_1, C_2):
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_head_i = self.bn_ent_i(self.emb_ent_i(e1_idx))
        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        emb_rel_i = self.bn_rel_i(self.emb_rel_i(rel_idx))

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)

        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and all entities.
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(b * emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(b * emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb, rel_emb

class Distmult(torch.nn.Module):
    def __init__(self, param):
        super(Distmult, self).__init__()
        self.name = 'Distmult'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']
        self.loss = torch.nn.BCELoss()
        # Real embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        # Real embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(self.param['input_dropout'])
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(self.param['input_dropout'])
        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        # (1.2) Real embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        real_score = torch.mm(emb_head_real * self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                              self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
        score = real_score
        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data, self.emb_rel_real.weight.data


class Tucker(torch.nn.Module):
    def __init__(self, param):
        super(Tucker, self).__init__()
        self.name = 'Tucker'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']

        self.E = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.R = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (self.embedding_dim, self.embedding_dim, self.embedding_dim)),
                         dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(self.param['input_dropout'])
        self.hidden_dropout1 = torch.nn.Dropout(self.param["hidden_dropout"])
        self.hidden_dropout2 = torch.nn.Dropout(self.param["hidden_dropout"])
        self.bn0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward_head_batch(self, e1_idx, rel_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(rel_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def get_embeddings(self):
        return self.E.weight.data, self.R.weight.data