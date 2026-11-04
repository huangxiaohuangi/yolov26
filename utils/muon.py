# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MuSGD optimizer (YOLO26 默认优化器)。

MuSGD = Muon + SGD 的混合优化器，与官方 `ultralytics/optim/muon.py` 的数学行为一致：

  * Muon 分量：对梯度做动量 EMA，再用 Newton-Schulz 五次迭代求极分解正交因子 UV^T，
    使更新矩阵的奇异值分布更均匀（各方向步长一致），按 ``lr * muon`` 施加；
  * SGD 分量：标准 momentum / Nesterov 更新，按 ``lr * sgd`` 施加。

官方在 `BaseTrainer.build_optimizer` 里把 ``ndim in {2, 4}`` 的参数（线性层权重与卷积核）
放进 Muon 参数组，其余（bias、BN 权重、1D 参数）走纯 SGD。本实现保持同样的分组语义，
只是额外提供一个 CPU 快速路径：在 CPU 上对 3x3/5x5 卷积核做 Newton-Schulz 的开销
远超前向本身，因此默认只对“线性等价”的权重（2D 权重 + 1x1 卷积）启用 Muon。

参考：
  - Muon: https://github.com/KellerJordan/Muon
  - YOLO26: https://docs.ultralytics.com/models/yolo26
"""

import torch
from torch import optim


def zeropower_via_newtonschulz5(G, eps=1e-7):
    """用 Newton-Schulz 五次迭代近似矩阵极分解的正交因子 UV^T。

    Args:
        G (torch.Tensor): 2D 矩阵或 3D 矩阵批次。
        eps (float): 数值稳定项。

    Returns:
        (torch.Tensor): 与 G 同形状的正交化结果（GPU 上用 bfloat16 计算）。
    """
    assert G.ndim in {2, 3}
    dtype = torch.bfloat16 if G.is_cuda else torch.float32  # bf16 在 GPU 上快得多，CPU 无 BF16 指令集
    X = G.reshape(-1, G.size(-2), G.size(-1)).to(dtype)
    X /= X.norm(dim=(-2, -1), keepdim=True) + eps  # 保证最大奇异值 <= 1
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)  # 让短边在最后，减少计算量
    a, b, c = 3.4445, -4.7750, 2.0315  # 五次多项式系数
    for _ in range(5):
        A = X @ X.transpose(-2, -1)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)  # b * A + c * A @ A
        X = torch.baddbmm(X, B, X, beta=a)  # a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.reshape(G.shape)


def muon_update(grad, momentum, beta=0.95, nesterov=True):
    """对一批梯度做「动量 EMA + Newton-Schulz 正交化」，返回 Muon 更新量。

    列数相同的矩阵会零填充后合并成一次批处理调用，动量计算用 foreach 融合算子，
    避免逐个参数启动 kernel。

    Args:
        grad (torch.Tensor | list[torch.Tensor]): 梯度（或梯度列表）。
        momentum (torch.Tensor | list[torch.Tensor]): 动量缓冲（原地更新）。
        beta (float): 动量系数。
        nesterov (bool): 是否使用 Nesterov 加速。

    Returns:
        (torch.Tensor | list[torch.Tensor]): 与输入同形状的更新量。
    """
    single = isinstance(grad, torch.Tensor)
    grads, momentums = ([grad], [momentum]) if single else (grad, momentum)
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)

    buckets = {}  # 按 (列数, 缩放, device, dtype) 分桶，便于批量正交化
    for i, u in enumerate(updates):
        m = u.flatten(1) if u.ndim > 2 else u.contiguous()
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(1), scale, m.device, m.dtype), []).append((i, m))

    for (cols, scale, device, dtype), items in buckets.items():
        items.sort(key=lambda t: -t[1].size(0))
        start = 0
        for j in range(1, len(items) + 1):  # 每组的行数跨度不超过 16 倍，限制填充浪费
            if j == len(items) or items[start][1].size(0) > 16 * items[j][1].size(0):
                group = items[start:j]
                X = torch.zeros(len(group), group[0][1].size(0), cols, device=device, dtype=dtype)
                torch._foreach_add_([X[k, : m.size(0)] for k, (_, m) in enumerate(group)],
                                    [m for _, m in group])
                X = zeropower_via_newtonschulz5(X).contiguous().to(dtype).mul_(scale)
                for k, (i, m) in enumerate(group):
                    updates[i] = X[k, : m.size(0)].reshape(grads[i].shape)
                start = j
    return updates[0] if single else updates


class MuSGD(optim.Optimizer):
    """Muon + SGD 混合优化器。

    Args:
        params (iterable): 参数或参数组。
        lr (float): 学习率。
        momentum (float): 动量系数（Muon 的 EMA 与 SGD 共用）。
        weight_decay (float): 权重衰减（只作用于 SGD 分量，与官方一致）。
        nesterov (bool): SGD 分量是否用 Nesterov。
        use_muon (bool): 该组是否启用 Muon 分量。
        muon (float): Muon 分量学习率系数（官方 0.2）。
        sgd (float): SGD 分量学习率系数（官方 1.0）。

    Notes:
        - ``use_muon=True`` 的参数组同时获得 Muon 与 SGD 两个更新；
        - ``use_muon=False`` 的参数组只做 SGD。
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False,
                 use_muon=False, muon=0.2, sgd=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            for p in params:
                if len(self.state[p]) == 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    if group["use_muon"]:
                        self.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)

            if group["use_muon"]:
                updates = muon_update(
                    [p.grad for p in params],
                    [self.state[p]["momentum_buffer"] for p in params],
                    beta=momentum,
                    nesterov=nesterov,
                )
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                buffers = [self.state[p]["momentum_buffer_SGD"] for p in params]
                lr *= self.sgd
            else:
                buffers = [self.state[p]["momentum_buffer"] for p in params]

            # SGD 分量（权重衰减只在这里生效）
            grads = [p.grad for p in params]
            if group["weight_decay"] != 0:
                grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)
            upd = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
            torch._foreach_add_(params, upd, alpha=-lr)
        return loss
