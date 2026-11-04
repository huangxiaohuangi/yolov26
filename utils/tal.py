# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Task-Aligned Assignment (TAL) 与 anchor/box 编解码工具.

YOLO26 沿用 YOLOv8 的 anchor-free + TAL 标签分配，并在此基础上加入
"STAL"（Small-Target-Aware Label Assignment）：
  1. 小于 stride 的 GT 框在选正样本时被放大到 stride 大小，保证小目标也能分到正样本；
  2. one2one 分支用 topk2 做二次筛选（默认 topk=7 -> topk2=1），得到唯一正样本，
     从而支持无 NMS 的端到端推理。
"""

import torch
import torch.nn as nn

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import bbox_iou


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor points and stride tensors from feature maps.

    Args:
        feats (list[torch.Tensor]): feature maps of each level, (b, c, h, w).
        strides (torch.Tensor | list): stride of each level.
        grid_cell_offset (float): 0.5 -> cell center.

    Returns:
        (tuple): anchor_points (N, 2) in grid units, stride_tensor (N, 1).
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    device = feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max=None):
    """Transform bbox(xyxy) to distance(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


class TaskAlignedAssigner(nn.Module):
    """Task-aligned assigner：按 (分类分^alpha * IoU^beta) 为每个 GT 选 topk 个 anchor。

    Args:
        topk (int): 每个 GT 选择的候选数。
        num_classes (int): 类别数。
        alpha (float): 分类权重。
        beta (float): 定位权重。
        stride (list): 各层 stride，用于 STAL 的小目标框放大。
        eps (float): 数值稳定项。
        topk2 (int | None): 二次 topk 筛选（STAL），None 表示不筛选。
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, stride=None, eps=1e-9, topk2=None):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]  # STAL: 小目标框放大的下限
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): (bs, na, nc) 已 sigmoid 的分类分。
            pd_bboxes (torch.Tensor): (bs, na, 4) 预测框 xyxy（像素）。
            anc_points (torch.Tensor): (na, 2) anchor 中心（像素）。
            gt_labels (torch.Tensor): (bs, n_max_boxes, 1)。
            gt_bboxes (torch.Tensor): (bs, n_max_boxes, 4) xyxy。
            mask_gt (torch.Tensor): (bs, n_max_boxes, 1)。

        Returns:
            (tuple): target_labels (bs, na), target_bboxes (bs, na, 4), target_scores (bs, na, nc),
                fg_mask (bs, na) bool, target_gt_idx (bs, na)。
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0]),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize（用每个 GT 的最大对齐度归一化，得到软标签分数）
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        overlaps *= mask_pos
        pos_overlaps = overlaps.amax(dim=-1, keepdim=True)  # b, max_num_obj
        align_metric.mul_(pos_overlaps).div_(pos_align_metrics + self.eps)
        norm_align_metric = align_metric.amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt.bool()
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric (score^alpha * iou^beta) given predicted and ground truth boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        shape = self.bs, self.n_max_boxes, na
        indices = mask_gt.nonzero(as_tuple=True)
        bbox_scores = pd_scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]
        overlaps = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        align_metric = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_scores.device)
        if indices[0].numel():  # 有候选时才计算 IoU，避免空张量报错
            overlap_values = self.iou_calculation(gt_bboxes[indices[:2]], pd_bboxes[indices[0], indices[2]])
            overlaps[indices] = overlap_values
            align_metric[indices] = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate CIoU for horizontal bounding boxes, clamped to be non-negative."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k candidates based on the given metrics (one anchor -> only one GT)."""
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # 统计每个 anchor 被选中的次数，>1 说明歧义，丢弃
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs, dtype=torch.int8)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones[:, :, k : k + 1])
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points."""
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)

        # 比 F.one_hot() 快 ~10x
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, nc)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = target_scores * (fg_mask[:, :, None] > 0)
        return target_labels, target_bboxes, target_scores.to(torch.float32)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """Select positive anchor centers within ground truth boxes (STAL: 小目标框放大到 stride)."""
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride_val  # 边长小于 stride 的小目标
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(float(self.stride_val), dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2)
        mask = xy_centers[:, 0] - lt[..., 0] > eps
        mask &= xy_centers[:, 1] - lt[..., 1] > eps
        mask &= rb[..., 0] - xy_centers[:, 0] > eps
        mask &= rb[..., 1] - xy_centers[:, 1] > eps
        return mask

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select the GT with highest IoU when an anchor is assigned to multiple GTs, then apply topk2 (STAL)."""
        fg_mask = mask_pos.sum(-2)
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.max(1).indices  # (b, h*w)
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)

        if self.topk2 != self.topk:  # one2one 分支：只保留对齐度最高的 topk2 个正样本
            align_metric = align_metric * mask_pos
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx

        fg_mask, target_gt_idx = mask_pos.max(-2)
        return target_gt_idx, fg_mask, mask_pos
