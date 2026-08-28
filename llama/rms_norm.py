import torch
import torch.nn as nn

from torch.nn import functional as F


#[batch_size, seq_len, d_model]
class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.a = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-9

    def forward(self, x):
        output = x / torch.sqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

        return output * self.a
        
# if __name__ == "__main__":
#     norm = RMSNorm(d_model=4096)
#     x = torch.randn(2, 128, 4096) # B=2, L=128
#     y = norm(x)
#     print(y.shape) # torch.Size([2, 128, 4096])
