# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Common modules for YOLO26 (YOLOv5 风格：一个文件放全部基础模块).

与 YOLOv5 的 models/common.py 对应，区别在于：
  * 检测头由 anchor-based(带 anchors、obj 分支) 改为 anchor-free + 双头(one2many / one2one)；
  * 新增 YOLO26 的 C3k2 / C2PSA / PSABlock 等模块；
  * reg_max=1 时不做 DFL，直接回归 ltrb 距离（YOLO26 的 "DFL removal"）。
"""

import ast
import math
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import make_divisible
from utils.tal import dist2bbox, make_anchors

__all__ = (
    "Attention",
    "Bottleneck",
    "C2PSA",
    "C3",
    "C3k",
    "C3k2",
    "Concat",
    "Conv",
    "DFL",
    "DWConv",
    "Detect",
    "PSABlock",
    "SPPF",
    "autopad",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm2d + activation (default SiLU)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation.

        Args:
            c1 (int): input channels.
            c2 (int): output channels.
            k (int): kernel size.
            s (int): stride.
            p (int | None): padding, auto-pad if None.
            g (int): groups.
            d (int): dilation.
            act (bool | nn.Module): activation, True -> default_act, False -> Identity.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization (after fuse)."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """Standard bottleneck (YOLO26 版本，kernel 支持 tuple)."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): input channels.
            c2 (int): output channels.
            shortcut (bool): use residual connection.
            g (int): groups.
            k (tuple): kernel sizes of cv1 and cv2.
            e (float): expansion ratio (hidden channels = c2 * e).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions (YOLOv5 CSP 结构)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with 3 convolutions."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with customizable kernel sizes (YOLO26 的 C3k)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initialize C3k module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(nn.Module):
    """YOLO26 主干/颈部核心模块：CSP + 可选的 C3k / PSABlock 内部块.

    yaml 参数: [c2, c3k(bool), e(float), attn(bool)]
      * c3k=False, attn=False -> 内部用 Bottleneck（等价 C2f 的快速版）
      * c3k=True              -> 内部用 C3k
      * attn=True             -> 内部用 Bottleneck + PSABlock（P5 层用注意力增强）
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        """Initialize C3k2 module."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C3k2 layer (split -> n blocks -> concat -> conv)."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        """Initialize the SPPF layer with given input/output channels and kernel size."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Attention(nn.Module):
    """Position-Sensitive Attention (PSA) 的自注意力模块（带位置编码的轻量 MHA）。"""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initialize multi-head attention module."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """Forward pass of the Attention module."""
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """Position-Sensitive Attention block: attention + FFN，均带残差。"""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        """Initialize the PSABlock."""
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Execute a forward pass through PSABlock."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """CSP + PSA：YOLO26 主干末端的全局注意力模块。"""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initialize C2PSA module."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        """Process the input tensor through a series of PSA blocks."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLO26 Concat module."""
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) 的积分层（YOLO26 默认 reg_max=1，不使用 DFL）。"""

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """YOLO26 检测头（anchor-free，可选 NMS-free 双头）。

    与 YOLOv5 的 Detect 区别：
      * 不再有 anchors / obj 分支，输出为 4*reg_max + nc；
      * reg_max=1 时直接回归 ltrb（无 DFL），reg_max>1 时保留 DFL 分支；
      * end2end=True 时额外挂一个 one2one 分支（topk=1 的标签分配），推理时直接输出 top-k 结果，
        无需 NMS（YOLO26 的 "NMS-free end-to-end"）。

    推理输出：
      * end2end=True : (bs, max_det, 6) -> [x1, y1, x2, y2, conf, cls]
      * end2end=False: (bs, num_anchors, 4 + nc) -> [xywh, cls_scores...]（配合 NMS 使用）
    """

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    max_det = 300  # max detections for end2end inference
    agnostic_nms = False

    def __init__(self, nc=80, reg_max=1, end2end=True, ch=()):
        """Initialize the YOLO26 detection head.

        Args:
            nc (int): number of classes.
            reg_max (int): DFL bins (1 -> direct ltrb regression, YOLO26 default).
            end2end (bool): build the extra one2one branch for NMS-free inference.
            ch (tuple): input channels of each detection level.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # 回归分支（box）
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # 分类分支（cls）：DWConv 分离卷积，轻量化
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.one2one_cv2 = nn.ModuleList()
        self.one2one_cv3 = nn.ModuleList()
        self._end2end = bool(end2end)
        if end2end:
            self.one2one_cv2 = deepcopy(self.cv2)
            self.one2one_cv3 = deepcopy(self.cv3)

    @property
    def end2end(self):
        """Whether the one2one (NMS-free) branch exists and is used at inference."""
        return self._end2end and len(self.one2one_cv2) > 0

    @end2end.setter
    def end2end(self, value):
        """Switch inference branch without touching the dual-head training."""
        if value and len(self.one2one_cv2) == 0:
            warnings.warn("This model has no one-to-one head; falling back to one-to-many outputs.")
            return
        self._end2end = bool(value)

    def forward_head(self, x, box_head=None, cls_head=None):
        """Run box/cls heads on the feature pyramid and return a flat prediction dict."""
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x):
        """Forward pass.

        Args:
            x (list[torch.Tensor]): feature maps of the detection levels.

        Returns:
            dict | tuple: 训练时返回 {'one2many': dict, 'one2one': dict}；
            推理时返回 ((bs, na, 4+nc) 或 (bs, max_det, 6), raw dict)。
        """
        preds = self.forward_head(x, self.cv2, self.cv3)  # one2many（主分支，训练/推理都可用）
        if len(self.one2one_cv2) > 0:
            # 训练时 detach，保证 one2one 分支不回传梯度到 backbone（避免双倍显存）
            x_detach = [xi.detach() for xi in x] if self.training else x
            preds = {"one2many": preds, "one2one": self.forward_head(x_detach, self.one2one_cv2, self.one2one_cv3)}
        if self.training:
            return preds
        y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
        y = y.permute(0, 2, 1)  # (bs, 4+nc, na) -> (bs, na, 4+nc)，与 YOLOv5 的推理输出对齐
        if self.end2end:
            y = self.postprocess(y)
        return y if self.export else (y, preds)

    def _inference(self, x):
        """Decode boxes and sigmoid the class scores, return (bs, 4+nc, na)."""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x):
        """Decode ltrb distances to boxes with anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes, anchors):
        """Decode distances to boxes: end2end 用 xyxy，普通模式用 xywh（配合 NMS）。"""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    def postprocess(self, preds):
        """Select top-k detections (NMS-free) -> (bs, max_det, 6) [x1, y1, x2, y2, conf, cls].

        与官方实现一致：先按每个 anchor 的最大类别分选出 top-k 个 anchor，再在其中取 top-k 个
        (anchor, class) 组合，这样每个 anchor 最多贡献若干个高分类别，且总数恒为 k。
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        k = min(self.max_det, scores.shape[1])  # anchors 数量通常 >> max_det
        _, ori_index = scores.max(dim=-1)[0].topk(k, dim=1)  # (bs, k) 候选 anchor
        cand = scores.gather(1, ori_index[..., None].expand(-1, -1, self.nc)).flatten(1)  # (bs, k*nc)
        s, index = cand.topk(k, dim=1)
        cls = (index % self.nc).unsqueeze(-1).float()
        idx = ori_index.gather(1, index // self.nc)  # 对应 anchor 下标
        boxes = boxes.gather(1, idx[..., None].expand(-1, -1, 4))
        return torch.cat([boxes, s.unsqueeze(-1), cls], dim=-1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls

    def fuse(self):
        """Drop the unused branch (inference only) to save a few ms."""
        end2end = self.end2end
        for name in tuple(self._modules):
            if name.startswith("one2one_") and not end2end:
                setattr(self, name, nn.ModuleList())
