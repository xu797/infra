import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from position_embedding import PositionEmbedding
from mutilhead_attention import MultiHeadAttention
from layer_norm import LayerNorm
from ffn import FeedForward

#input:[batch_size, seq_len, d_model]
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.mutilhead_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model)

    def forward(self, x, mask = None):

        atten_x, _ = self.mutilhead_attention(x, mask)
        x = self.layer_norm1(x + atten_x)
        ffn_x = self.ffn(x)
        x = self.layer_norm2(x + ffn_x)

        return x


    

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.pos_encoding = PositionEmbedding(d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(d_model, vocab_size)  # 最终输出层
        
    def forward(self, x, mask = None):
        # 1. 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        # 2. 多层编码器块处理
        for block in self.encoder_blocks:
            x = block(x)
            
        # 3. 最终线性层（类似linear+sigmoid的输出层）
        output = self.final_layer(x)
        return output
    
def generate_causal_mask(seq_len):
    # triu(diagonal=1)：对角线以上全部为True，代表要mask掉未来位置
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    # 升维 [1,1,S,S]，方便广播到 [B,num_heads,S,S]
    return mask.unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":
    vocab_size = 100
    model = SimpleTransformer(vocab_size=vocab_size, d_model=16, num_heads=4, num_layers=2)
    B, S = 2, 5
    x = torch.randint(0, vocab_size, (B, S))

    # 生成causal mask，GPT风格
    mask = generate_causal_mask(S)
    logits = model(x, mask=mask)
    print(logits.shape) # [2,5,100]

