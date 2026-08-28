import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 1000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        #x_shape:[batch_size, seq_len, d_model]
        #pe_shape:[batch_size, max_seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

# if __name__ == "__main__":
#     pe_layer = PositionEmbedding(d_model=8)
#     input = torch.randn(2, 4, 8)
#     print(input.shape)
#     print(input)
#     out = pe_layer(input)
#     print(out.shape)
#     print(out)
