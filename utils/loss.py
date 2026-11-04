# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for YOLO26.

与 YOLOv5 的 ComputeLoss（anchor-based + obj 分支）不同，YOLO26 使用：
  * anchor-free 的 TAL 标签分配；
  * box 回归：CIoU + （reg_max=1 时）归一化 L1，代替 DFL；
  * 双头训练：one2many(topk=10) + one2one(topk=7 -> topk2=1)，
    并按 ProgLoss 的思路让 one2many 的权重随 epoch 从 0.8 线性衰减到 0.1。
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL), only used when reg_max > 1."""

    def __init__(self, reg_max=16):
        """Initialize the DFL module with the number of distribution bins."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://arxiv.org/abs/2006.04388."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        logp = F.log_softmax(pred_dist, dim=1)
        return -(
            logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tr.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=1):
        """Initialize the BboxLoss module; DFL is used when reg_max > 1, otherwise a normalized L1 loss."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    @staticmethod
    def _normalize_ltrb(x, stride, imgsz):
        """Scale ltrb distances by stride and normalize by image size (YOLO26 的 L1 回归目标)."""
        x = x * stride
        sx, sy = imgsz[1], imgsz[0]
        return torch.stack((x[..., 0] / sx, x[..., 1] / sy, x[..., 2] / sx, x[..., 3] / sy), dim=-1)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        """Compute CIoU and DFL (or L1 when reg_max <= 1) losses for bounding boxes."""
        fg_mask = fg_mask.nonzero(as_tuple=True)
        if fg_mask[0].numel() == 0:  # 空前景：返回 0，同时保持计算图（pred_dist 参与 0 乘）
            zero = pred_dist.sum() * 0.0
            return zero, zero

        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:  # YOLO26: 直接回归归一化后的 ltrb（L1）
            target_ltrb = self._normalize_ltrb(bbox2dist(anchor_points, target_bboxes), stride, imgsz)
            pred_ltrb = self._normalize_ltrb(pred_dist, stride, imgsz)
            loss_dfl = F.l1_loss(pred_ltrb[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Criterion class for computing detection loss (anchor-free + TAL), 对应 YOLOv5 的 ComputeLoss。

    Args:
        model (nn.Module): de-paralleled YOLO26 model (needs `model.model[-1]` Detect and `model.hyp`).
        tal_topk (int): TAL 每个 GT 的候选数。
        tal_topk2 (int | None): STAL 二次筛选（one2one 分支用 1）。
    """

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        """Initialize the loss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss") if self.use_dfl else ("box_loss", "cls_loss", "l1_loss")

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Pad targets into a (batch_size, max_targets, 5) tensor of [cls, x1, y1, x2, y2] in pixels.

        Args:
            targets (torch.Tensor): (N, 6) = [img_idx, cls, x, y, w, h] (normalized xywh).
            batch_size (int): batch size.
            scale_tensor (torch.Tensor): [w, h, w, h] of the input image size.
        """
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        counts = torch.bincount(targets[:, 0].long(), minlength=batch_size)
        n_max = int(counts.max())
        out = torch.zeros(batch_size, n_max, 5, device=self.device)
        for b in range(batch_size):
            idx = targets[:, 0].long() == b
            if idx.any():
                t = targets[idx, 1:].clone()  # cls, x, y, w, h
                t[:, 1:5] = xywh2xyxy(t[:, 1:5] * scale_tensor)
                out[b, : t.shape[0]] = t
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds, batch):
        """Calculate the box, cls and l1/dfl losses and return them with the assigned targets."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets -> (bs, n_max, 5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)  # 避免除 0

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores,
            target_scores_sum, fg_mask, imgsz, stride_tensor,
        )

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp.get("dfl", 1.5)  # dfl / l1 gain
        return (fg_mask, target_bboxes, anchor_points, stride_tensor), loss, dict(zip(self.loss_names, loss.detach()))

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and l1/dfl multiplied by batch size."""
        return self.loss(preds, batch)

    def loss(self, preds, batch):
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        _, loss, loss_items = self.get_assigned_targets_and_loss(preds, batch)
        return loss * batch_size, loss_items  # for backward


class E2ELoss:
    """双头训练损失：one2many + one2one，权重按 ProgLoss 思路随 epoch 衰减。

    Args:
        model (nn.Module): YOLO26 model.
        loss_fn (type): 单分支损失类，默认 v8DetectionLoss。
    """

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        # ProgLoss: one2many 权重 0.8 -> 0.1 线性衰减，one2one 权重 0.2 -> 0.9
        self.o2m = 0.8
        self.o2o = 0.2
        self.start_o2m = 0.8
        self.final_o2m = 0.1
        self.epoch = 0
        self.epochs = int(getattr(model, "hyp", {}).get("epochs", 100)) if hasattr(model, "hyp") else 100
        self.hyp = self.one2one.hyp

    def set_epoch(self, epoch, epochs=None):
        """Update the one2many / one2one loss weights for the given epoch."""
        self.epoch = epoch
        if epochs:
            self.epochs = epochs
        frac = epoch / max(self.epochs - 1, 1)  # 0 -> 1
        self.o2m = max(1 - frac, 0) * (self.start_o2m - self.final_o2m) + self.final_o2m
        self.o2o = max(1.0 - self.o2m, 0)

    def __call__(self, preds, batch):
        """Calculate the weighted one-to-many plus one-to-one loss (loss items 取 one2one 分支)."""
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many, _ = self.one2many.loss(one2many, batch)
        loss_one2one, items = self.one2one.loss(one2one, batch)
        return loss_one2many * self.o2m + loss_one2one * self.o2o, items


class ComputeLoss:
    """对外统一的损失入口（YOLOv5 里同名类），根据模型是否 end2end 自动选择损失。

    Usage:
        compute_loss = ComputeLoss(model)
        loss, loss_items = compute_loss(pred, targets)  # pred 为模型 train 模式的输出
    """

    def __init__(self, model):
        """Initialize ComputeLoss."""
        self.e2e = getattr(model.model[-1], "end2end", False)
        self.criterion = E2ELoss(model) if self.e2e else v8DetectionLoss(model, tal_topk=10)
        self.hyp = getattr(model, "hyp", {})

    def set_epoch(self, epoch, epochs=None):
        """Update loss weights at the beginning of each epoch."""
        if hasattr(self.criterion, "set_epoch"):
            self.criterion.set_epoch(epoch, epochs)

    def __call__(self, preds, targets):
        """Calculate the loss.

        Args:
            preds (dict | tuple): model output in train mode.
            targets (torch.Tensor): (N, 6) = [img_idx, cls, x, y, w, h] (normalized).

        Returns:
            (tuple): (scalar loss (已按 batch_size 缩放并求和), loss_items dict)
        """
        preds = preds[1] if isinstance(preds, tuple) else preds
        batch = {
            "batch_idx": targets[:, 0],
            "cls": targets[:, 1],
            "bboxes": targets[:, 2:6],
        }
        loss, items = self.criterion(preds, batch)
        return loss.sum(), items  # 与 YOLOv5 一致：返回标量 loss 供 backward
