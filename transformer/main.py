import torch
import torch.nn as nn

import math


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        d_k = self.d_model // self.num_heads

        Q = Q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

        atten_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        atten_weight = torch.softmax(atten_score, -1)

        #[batch_size, num_heads, seq_len, d_k]
        out = torch.matmul(atten_weight, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.w_o(out)

        return out, atten_weight


if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    print(x.shape)

    self_atten = SelfAttention(8)

    print(self_atten(x))