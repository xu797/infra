# infra

1. **什么情况下需要 `keepdim`？**
做 `mean / sum / std` 这类维度聚合运算时使用。开启后会保留被压缩的维度，方便后续广播运算。
典型例子：手写 LayerNorm：`x.mean(-1, keepdim=True)`。如果不开启，张量会丢失一维，`x‑mean` 广播计算直接报错。

2. **什么情况下需要 `contiguous()`？**
张量经过 `transpose()` / `permute()` 维度置换之后，**逻辑维度改变，但内存存储不再连续**。
在调用 `view()` 做形状重塑之前，必须调用 `.contiguous()`，复制得到内存连续的张量，否则抛出 RuntimeError。
> 注意：`reshape()` 内部会自动处理连续性，但工程代码中多头注意力等场景习惯显式写 `.contiguous()`。

3. **什么情况下需要用 `nn.Parameter`？**
将普通 `torch.Tensor` 注册为模型**可训练参数**。
放在 `nn.Module` 内部，会自动加入 `model.parameters()`，参与反向传播与权重更新。
手写 LayerNorm 里面的缩放 $\gamma$、偏移 $\beta$（`a_1` / `b_1`）就需要用 `nn.Parameter`。
> ⚠️注意：类名是大写 `nn.Parameter`，小写 `nn.parameter` 会直接报错。
