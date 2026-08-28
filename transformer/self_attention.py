import torch
import torch.nn as nn
from torch.nn import functional as F

import math


#input:[batch_size, seq_len, d_model]
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)


    def forward(self, x):

        Q = self.w_q(x)
        # print(Q.shape)
        K = self.w_k(x)
        # print(K.shape)
        V = self.w_v(x)
        # print(V.shape)

        # Q @ K^T 点积注意力分数

        atten_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        # print(atten_score.shape)
        # print(atten_score)
        #soft_max
        attn_weight = torch.softmax(atten_score, dim=-1) # 注意力权重
        out = torch.matmul(attn_weight, V)  # [B,S,D]
        return out, attn_weight

# if __name__ == "__main__":
#     d_model = 8
#     # pe_layer = PositionEmbedding(d_model)
#     attn = SelfAttention(d_model)

#     x = torch.randn(2,4,8)   # [B=2, S=4, D=8]
#     # x = pe_layer(x)          # 加上位置编码

#     out, attn_w = attn(x)
#     # print("out shape:", out.shape)      # torch.Size([2,4,8])
#     # print("attn weight shape:", attn_w.shape) # torch.Size([2,4,4])