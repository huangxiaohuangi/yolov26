# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PyTorch utils: device selection, weight init, EMA, model info, optimizers."""

import math
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn

from utils.general import LOGGER, colorstr

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


def select_device(device="", batch=0, newline=False, verbose=True):
    """Selects the appropriate PyTorch device based on the provided arguments.

    Args:
        device (str): 'cpu', 'mps', '0', '0,1,2,3' or '' (auto-select).
        batch (int): batch size used in the log message.
        newline (bool): add a newline to the log message.
        verbose (bool): print the device information.

    Returns:
        (torch.device): selected device.
    """
    s = f"Ultralytics YOLO26 🚀 Python-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # remove sequential commas, i.e. "0,1,2,3"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f"Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f"i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else ["0"]  # i.e. ['0', '1'] for device='0,1'
        n = len(devices)  # number of devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        device = torch.device("cuda:0")
        if verbose:
            LOGGER.info(f"{s}CUDA ({n} devices) ready")
    elif mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            LOGGER.info(f"{s}MPS ready")
    else:  # revert to CPU
        device = torch.device("cpu")
        if verbose:
            LOGGER.info(f"{s}CPU ready")

    if newline:
        LOGGER.info("")
    return device


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def initialize_weights(model):
    """Initialize model weights to random values (YOLO convention: BN gamma=1, beta=0)."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales and pads an image tensor of shape (B, C, H, W), optionally preserving shape."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = torch.nn.functional.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return torch.nn.functional.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Print model information: layers, parameters, gradients and (optionally) GFLOPs."""
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    flops = "?"
    if thop:  # FLOPs
        try:
            im = torch.zeros(1, 3, imgsz, imgsz).to(next(model.parameters()).device)
            flops = f"{thop.profile(model, inputs=(im,), verbose=False)[0] / 1e9 * 2:.1f} GFLOPs"
        except Exception:
            flops = "?"
    LOGGER.info(
        f"Model summary: {n_l} layers, {n_p:,} parameters, {n_g:,} gradients, {flops}"
    )

    if detailed:
        LOGGER.info(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        for i, (f, n, m, args) in enumerate(getattr(model, "layer_records", [])):
            LOGGER.info(f"{i:>3}{f!s:>20}{n:>3}{sum(x.numel() for x in m.parameters()):10.0f}  "
                        f"{str(type(m))[8:-2].replace('__main__.', ''):<45}{args!s:<30}")


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def profile(input, ops, n=10, device=None):
    """YOLO speed/memory/FLOPs profiler.

    Args:
        input (torch.Tensor): input tensor, e.g. torch.randn(16, 3, 640, 640).
        ops (nn.Module | list): model(s) to profile.
        n (int): number of iterations.
        device (torch.device): device to profile on.

    Returns:
        (list): profile results, one dict per op.
    """
    results = []
    device = device or select_device()
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2 if thop else 0
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f"{p:12,}{flops:12.2f}{mem:14.2f}{tf:14.1f}{tb:14.1f}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models."""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with decay 'decay', 'tau' and 'updates'."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters."""
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Update EMA attributes by copying named attributes from model."""
        copy_attr(self.ema, model, include, exclude)


class EarlyStopping:
    """Early stopping class that stops training when a monitored metric has stopped improving."""

    def __init__(self, patience=30):
        """Initialize early stopping object.

        Args:
            patience (int): Number of epochs to wait after last time monitored metric improved.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """Check whether to stop training."""
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                        f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                        f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                        f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.")
        return stop


def smart_optimizer(model, name="MuSGD", lr=0.001, momentum=0.9, decay=1e-5, muon_scope="auto"):
    """Construct an optimizer with weight decay for PyTorch models, YOLOv5 style.

    参数分组语义与官方 `BaseTrainer.build_optimizer` 对齐：
      * ``MuSGD`` 时额外拆出一个 Muon 组（``ndim in {2, 4}`` 的权重矩阵 / 卷积核），
        其余为 weight(带 decay) / bn weight(无 decay) / bias(无 decay) 三组；
      * 分类头 ``Detect.cv3`` 的参数使用 ``lr * 3``（官方 YOLO26 的微调策略）。

    Args:
        model (nn.Module): model to optimize.
        name (str): 'MuSGD', 'SGD', 'Adam', 'AdamW' 或 'NAdam'。
        lr (float): initial learning rate.
        momentum (float): momentum / beta1.
        decay (float): weight decay.
        muon_scope (str): 'auto' | 'all' | 'linear'。'auto' 在 CPU 上只对 2D 权重与 1x1 卷积
            启用 Muon（3x3 卷积核的 Newton-Schulz 在 CPU 上比前向还慢），在 CUDA 上等价于 'all'。

    Returns:
        (torch.optim.Optimizer): optimizer instance.
    """
    from utils.muon import MuSGD  # local import to avoid circular deps

    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层类型
    use_muon = name == "MuSGD"
    if muon_scope == "auto":
        muon_scope = "all" if next(model.parameters()).is_cuda else "linear"

    def is_muon(p):
        """判断某个权重是否进入 Muon 参数组。"""
        if not use_muon or p.ndim not in {2, 4}:
            return False
        if muon_scope == "all" or p.ndim == 2:
            return True
        return p.shape[2] == 1 and p.shape[3] == 1  # 1x1 卷积（线性等价）

    g_w, g_bnw, g_b, g_muon = [], [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g_b.append(v.bias)
        if isinstance(v, bn):
            if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                g_bnw.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            (g_muon if is_muon(v.weight) else g_w).append(v.weight)

    # 分类头（Detect.cv3 / one2one_cv3）参数用更高学习率
    head = model.model[-1] if hasattr(model, "model") and len(model.model) else None
    boosted = set()
    for attr in ("cv3", "one2one_cv3"):
        m = getattr(head, attr, None)
        if m is not None:
            boosted |= {id(p) for p in m.parameters()}

    def split(params):
        """把参数按是否属于分类头拆成 (boosted, normal) 两个列表。"""
        return [p for p in params if id(p) in boosted], [p for p in params if id(p) not in boosted]

    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        opt_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
    elif name == "RMSprop":
        opt_args = {"lr": lr, "momentum": momentum}
    elif name in {"SGD", "MuSGD"}:
        opt_args = {"lr": lr, "momentum": momentum, "nesterov": True}
    else:
        raise ValueError(f"Optimizer {name} not recognized, supported: MuSGD, SGD, Adam, AdamW, NAdam")

    groups = [
        {"params": g_w, **opt_args, "weight_decay": decay, "param_group": "weight"},
        {"params": g_bnw, **opt_args, "weight_decay": 0.0, "param_group": "bn"},
        {"params": g_b, **opt_args, "weight_decay": 0.0, "param_group": "bias"},
    ]
    if use_muon:
        groups.append({"params": g_muon, **opt_args, "weight_decay": decay,
                       "use_muon": True, "param_group": "muon"})

    # 拆出分类头参数组（lr*3）
    expanded = []
    for x in groups:
        x = dict(x)
        p = x.pop("params")
        p1, p2 = split(p)
        if p1:
            expanded.append({"params": p1, **x, "lr": lr * 3})
        expanded.append({"params": p2, **x})

    cls = (lambda params, **kw: MuSGD(params, muon=0.2, sgd=1.0, **kw)) if use_muon else getattr(torch.optim, name)
    optimizer = cls(expanded)

    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with "
                f"{len(expanded)} parameter groups: {len(g_w)} weight(decay={decay}), "
                f"{len(g_bnw)} bn(no decay), {len(g_b)} bias(no decay)"
                + (f", {len(g_muon)} muon(scope={muon_scope})" if use_muon else ""))
    return optimizer
