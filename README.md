# infra

**1) 什么情况下需要keepdim?**
做`mean/sum/std`等聚合运算；保留被压缩的维度，方便后续广播，像你手写 LayerNorm `x.mean(-1, keepdim=True)`，否则 shape 会少一维无法做`x‑mean`广播。
**2) 什么情况下需要contiguous?**
当张量经过 transpose/permute 后，逻辑 shape 和内存存储不连续；做 view/reshape 之前必须调用`.contiguous()`，否则报错。
**3) 什么情况下需要用nn.parameter?**
把普通 tensor 注册成模型可训练参数；放进`nn.Module`里面，会自动加入`model.parameters()`，参与反向传播更新权重，手写 LayerNorm 的 γ、β 就需要它。