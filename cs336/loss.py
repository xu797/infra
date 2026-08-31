import torch
import torch.nn as nn

from torch.nn import functional as F

#input:[batch_size, seq_len, vocab_size]
#groud_truth:[batch_size, seq_len]
def cross_entropy(logits, targets):
    #max_value
    #m.shape:[batch_size, seq_len, 1]
    m = torch.max(logits, dim=-1, keepdim=True).values

    #[batch_size, seq_len, vocab_size]
    shift_targets = logits - m

    #log_sum.shape:[batch_size, seq_len]
    log_sum = torch.log(torch.sum(torch.exp(shift_targets), dim=-1))

    groud_truth = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    
    loss_sum = log_sum + m.squeeze(-1) - groud_truth.squeeze(-1)

    return torch.mean(loss_sum)



# B, S, V = 2,3,5
# logits = torch.randn(B,S,V)
# targets = torch.randint(0, V, (B,S))

# loss_my = cross_entropy(logits, targets)
# loss_ref = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

# print(f"自己实现 loss: {loss_my.item():.6f}")
# print(f"官方F.cross_entropy loss: {loss_ref.item():.6f}")
# print(f"是否相等: {torch.allclose(loss_my, loss_ref)}")


