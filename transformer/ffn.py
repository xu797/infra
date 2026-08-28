import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = 2048):
        super().__init__()

        # self.d_model = d_model
        # self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        return x


