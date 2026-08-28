import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.a_1 = nn.Parameter(torch.ones(d_model))
        self.b_1 = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-9

    def forward(self, x):
        u = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        return self.a_1 * ((x - u) / (std + self.eps)) + self.b_1