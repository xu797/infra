import torch
import torch.nn as nn
from torch.nn import functional as F

#feedforward:[batch_size, seq_len, d_model]
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, self.d_ff)
        self.w2 = nn.Linear(self.d_ff, d_model)
        self.w3 = nn.Linear(d_model, self.d_ff)

    def forward(self, x):
        gate = self.w1(x)
        gate = gate*torch.Sigmoid(gate)

        value = self.w3(x)

        return self.w2(gate*value)
