import torch
from typing import Optional, Dict

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight: Optional[Dict[int, torch.tensor]]=None):
        super().__init__()
        self.weight = weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, inp, target):
        if self.weight:
            try:
                return (torch.Tensor(list(map(lambda x: self.weight[int(x)], target))).to(self.device) * (inp-target)**2).mean()
            except KeyError:
                return torch.tensor(10000.)
        return ((inp-target)**2).mean()
