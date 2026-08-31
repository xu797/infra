import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta_base = 10000.0, device=None):
        super().__init__()
        self.head_dim = head_dim

        powers = torch.arange(0, head_dim, 2, device=device).float() / head_dim
        freq = 1.0 / (theta_base ** powers)

        positions = torch.arange(0, max_seq_len, device=device).float()
        angles = torch.outer(positions, freq)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, token_position):
        """
        Args:
            x: 张量，可以是 [B, H, T, head_dim] / [B, T, head_dim]
            token_position: torch.LongTensor, shape [B, T], 每个token的位置编号
        Returns:
            rotated x，shape同输入
        """
        cos = self.cos_cached[token_position]   # [B, T, head_dim//2]
        sin = self.sin_cached[token_position]

        # 适配多头：如果x比cos多一维(head维)，插入维度用于广播
        if x.ndim > cos.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 对齐数据类型，支持fp16/bf16
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos
        return output


# if __name__ == "__main__":
#     # 测试4维多头输入 [B,H,T,head_dim]
#     B, H, T, head_dim = 2,8,64,128
#     rope = RoPE(head_dim=head_dim, max_seq_len=1024)

#     x = torch.randn(B, H, T, head_dim)
#     token_pos = torch.arange(T).expand(B,-1) # shape [B,T]

#     out = rope(x, token_pos)
#     print(out.shape) # torch.Size([2, 8, 64, 128])
