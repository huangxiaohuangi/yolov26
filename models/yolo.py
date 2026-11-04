# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26 model builder (YOLOv5 风格：yaml -> nn.Sequential).

Usage:
    $ python models/yolo.py --cfg yolo26n.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO26 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (  # noqa: E402
    C2PSA,
    C3,
    C3k,
    C3k2,
    SPPF,
    Bottleneck,
    Concat,
    Conv,
    DWConv,
    Detect,
)
from utils.general import LOGGER, check_yaml, colorstr, make_divisible, print_args  # noqa: E402
from utils.torch_utils import (  # noqa: E402
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    select_device,
)

__all__ = ("Detect", "DetectionModel", "Model", "parse_model")


class BaseModel(nn.Module):
    """YOLO26 base model (forward / fuse / info)."""

    def forward(self, x, profile=False):
        """Forward pass of the model (single-scale)."""
        return self._forward_once(x, profile)  # single-scale inference, train

    def _forward_once(self, x, profile=False):
        """Forward pass through all layers, honoring the 'from' (f) indices."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profile one layer: params and time."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        t = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if t:
            t.record()
        else:
            import time

            t0 = time.time()
        for _ in range(10):
            m(x.copy() if c else x)
        if t:
            t.record()
            torch.cuda.synchronize()
            dt.append(t.elapsed_time(t) / 10)
        else:
            dt.append((time.time() - t0) * 100)
        np_ = sum(x.numel() for x in m.parameters())
        LOGGER.info(f"{dt[-1]:10.2f} {'-':>10s} {np_:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuse Conv2d() and BatchNorm2d() layers for faster inference."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Print model information."""
        model_info(self, detailed=verbose, imgsz=img_size)

    def _apply(self, fn, recurse=True):
        """Apply transformations (to(), cpu(), cuda(), half()) including the Detect strides."""
        self._apply_done = True
        super()._apply(fn, recurse)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect) and m.stride is not None:
            m.stride = fn(m.stride)
        return self


class DetectionModel(BaseModel):
    """YOLO26 detection model: build from a yaml config."""

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO26 detection model with a yaml config, input channels and number of classes.

        Args:
            cfg (str | dict): model yaml path or dict.
            ch (int): input channels.
            nc (int): number of classes, overrides the yaml value when given.
            verbose (bool): print the layer table.
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            self.yaml_file = "model.yaml"
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # 支持 ultralytics 的 scales 写法：yolo26n.yaml -> scale 'n'
        if "scales" in self.yaml and "depth_multiple" not in self.yaml:
            scale = Path(self.yaml_file).stem[-1]  # i.e. 'n' from 'yolo26n'
            assert scale in self.yaml["scales"], f"Scale '{scale}' not in {list(self.yaml['scales'])}"
            gd, gw, mc = self.yaml["scales"][scale]
            self.yaml.update(depth_multiple=gd, width_multiple=gw, max_channels=mc)
        # 记录尺度名（m/l/x 的 C3k2 强制使用 C3k 内部块，与官方实现保持一致）
        self.yaml["scale"] = Path(self.yaml_file).stem[-1]

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides（anchor-free：用特征图尺寸反推 stride，不需要 anchors）
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            with torch.no_grad():
                feats = self._forward_once(torch.zeros(1, ch, s, s))  # train 模式下返回 dict
                feats = feats["one2many"]["feats"] if isinstance(feats, dict) else feats
                m.stride = torch.tensor([s / x.shape[-2] for x in feats])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info(verbose=verbose)
        if verbose:
            LOGGER.info("")

    def forward(self, x, profile=False):
        """Forward pass (train: dict of raw predictions; eval: (decoded preds, raw dict))."""
        return self._forward_once(x, profile)


Model = DetectionModel  # retain YOLO 'Model' class for backwards compatibility


def parse_model(d, ch, verbose=True):
    """Parse a YOLO26 model.yaml into an nn.Sequential + savelist.

    Args:
        d (dict): model dict.
        ch (list[int]): input channels, i.e. [3].
        verbose (bool): print the layer table.

    Returns:
        (tuple): nn.Sequential model, sorted list of layers whose outputs must be saved.
    """
    nc, gd, gw, act, ch_mul = (
        d["nc"],
        d.get("depth_multiple", 1.0),
        d.get("width_multiple", 1.0),
        d.get("activation"),
        d.get("channel_multiple"),
    )
    max_channels = d.get("max_channels", None)
    scale = d.get("scale", "")  # 尺度名 n/s/m/l/x
    reg_max = d.get("reg_max", 1)  # DFL bins, YOLO26 uses 1 (no DFL)
    end2end = d.get("end2end", True)  # NMS-free dual head
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    if not ch_mul:
        ch_mul = 8

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    layers, save, c2, records = [], [], ch[-1], []  # layers, savelist, ch out, layer table
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError, SyntaxError, TypeError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {Conv, DWConv, Bottleneck, C3, C3k, C3k2, C2PSA, SPPF}:
            c1, c2 = ch[f], args[0]
            if max_channels:  # 限制最大通道数（YOLO26 scales 的第三个参数）
                c2 = min(c2, max_channels)
            c2 = make_divisible(c2 * gw, ch_mul)  # 宽度缩放
            args = [c1, c2, *args[1:]]
            if m in {C3, C3k, C3k2, C2PSA}:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2 and scale in {"m", "l", "x"}:  # 官方实现：m/l/x 的 C3k2 使用 C3k 内部块
                args[3:4] = [True]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect}:
            args.extend([reg_max, end2end, [ch[x] for x in f]])
        elif m is nn.Upsample:
            c2 = ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>18}{n_:>3}{np:10.0f}  {t:<40}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        records.append((f, n_, m_, args))
        if i == 0:
            ch = []
        ch.append(c2)
    model = nn.Sequential(*layers)
    model.layer_records = records  # for model.info(verbose=True)
    return model, sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolo26n.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        from utils.torch_utils import profile as profile_fn

        results = profile_fn(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo26*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                LOGGER.error(f"in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
