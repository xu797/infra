import torch
import torch.nn as nn
from torch.nn import functional as F

import math


#input:[batch_size, seq_len, d_model]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须可以被num_heads整除"

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        #projection
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        #[batch_size, self.num_heads, seq_len, self.d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        #atten_score Q@K^T
        atten_score  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #soft_max
        atten_weight = torch.softmax(atten_score, -1)
        #output
        out = torch.matmul(atten_weight, V) #[batch_size, self.num_heads, seq_len, self.d_k]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        out = self.w_o(out)

        return out, atten_weight

# if __name__ == "__main__":
#     d_model = 16
#     num_heads = 4
#     mha = MultiHeadAttention(d_model, num_heads)

#     x = torch.randn(2, 5, d_model) # B=2, S=5

#     # 构造因果mask（解码器用，屏蔽未来token）
#     seq_len = 5

#     out, attn_w = mha(x)
#     print("out shape:", out.shape)         # torch.Size([2,5,16])
#     print("attn_weight shape:", attn_w.shape) # torch.Size([2,4,5,5])