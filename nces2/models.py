import torch, torch.nn as nn, numpy as np
from .modules import *

class SetTransformer(nn.Module):
    def __init__(self, vocab, num_examples, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.kwargs = kwargs
        self.num_examples = num_examples
        self.max_len = kwargs.max_length
        self.proj_dim = kwargs.proj_dim
        self.embedding_dim = kwargs.embedding_dim
        self.ln = kwargs.ln
        self.num_heads = kwargs.num_heads
        self.num_inds = kwargs.num_inds
        self.num_seeds = kwargs.num_seeds
        self.inv_vocab = np.array(vocab, dtype='object')
        self.vocab = {vocab[i]:i for i in range(len(vocab))}
        self.loss = nn.CrossEntropyLoss()
        
        self.enc = nn.Sequential(
                ISAB(self.embedding_dim, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln),
                ISAB(self.proj_dim, self.proj_dim, self.num_heads, self.num_inds, ln=self.ln))
        self.dec = nn.Sequential(
                PMA(self.proj_dim, self.num_heads, self.num_seeds, ln=self.ln),
                nn.Linear(self.proj_dim, len(self.vocab)*self.max_len))

    def forward(self, x1, x2):
        x1 = self.enc(x1)
        x2 = self.enc(x2)
        x = torch.cat([x1,x2], -2)
        x = self.dec(x).reshape(-1, len(self.vocab), self.max_len)
        aligned_chars = self.inv_vocab[x.argmax(1).cpu()]
        return aligned_chars, x
