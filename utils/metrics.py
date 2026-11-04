# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics: IoU, mAP, confusion matrix, fitness."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    """Return fitness as weighted combination of metrics: [P, R, mAP@0.5, mAP@0.5:0.95]."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """Compute the average precision, given the recall and precision curves.

    Args:
        tp (np.ndarray): True positives, shape (nc, n_preds) or (n_preds,).
        conf (np.ndarray): Confidence, shape (nc, n_preds) or (n_preds,).
        pred_cls (np.ndarray): Predicted classes.
        target_cls (np.ndarray): Ground-truth classes.
        plot (bool): Whether to plot P, R, F1 curves.
        save_dir (str): Directory to save plots.
        names (tuple): Class names.
        eps (float): Small value to avoid division by zero.
        prefix (str): Prefix for saved plot filenames.

    Returns:
        (tuple): tp, fp, p, r, f1, ap, ap_class, (px, py) curves.
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), (px, py)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves."""
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") else np.trapz(
            np.interp(x, mrec, mpre), x
        )  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool): The format of box coordinates. If True, boxes are in (x_center, y_center, width, height)
            format. If False, boxes are in (x1, y1, x2, y2) format.
        GIoU (bool): If True, calculate Generalized IoU.
        DIoU (bool): If True, calculate Distance IoU.
        CIoU (bool): If True, calculate Complete IoU.
        eps (float): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    """Calculate intersection-over-union (IoU) of boxes, both in (x1, y1, x2, y2) format.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float): A small value to avoid division by zero.

    Returns:
        (tuple): pairwise IoU (N, M) and union area.
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps), inter


def bbox_ioa(box1, box2, eps=1e-7):
    """Calculate the intersection over box2 area given box1 and box2 (x1, y1, x2, y2)."""
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=(), xlabel="Recall", ylabel="Precision"):
    """Plot a precision-recall curve, saving to 'save_dir'."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    plt.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir="mc_curve.png", names=(), xlabel="Confidence", ylabel="Metric"):
    """Plot a metric-confidence curve, saving to 'save_dir'."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    plt.savefig(save_dir, dpi=250)
    plt.close()


class ConfusionMatrix:
    """Confusion matrix for object detection (background as extra class).

    Args:
        nc (int): number of classes.
        conf (float): confidence threshold.
        iou_thres (float): IoU threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initialize the confusion matrix."""
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """Update the confusion matrix with a batch of detections and labels.

        Args:
            detections (np.ndarray): (N, 6) [x1, y1, x2, y2, conf, cls].
            labels (np.ndarray): (M, 5) [cls, x1, y1, x2, y2].
        """
        if detections is None:
            gt_classes = labels[:, 0].astype(int)
            try:
                self.matrix[self.nc, gt_classes] += 1
            except IndexError:
                pass
            return

        det_classes = detections[:, 5].astype(int)
        if labels.shape[0] == 0:
            self.matrix[det_classes, self.nc] += 1
            return

        iou = box_iou(torch.from_numpy(labels[:, 1:]).float(), torch.from_numpy(detections[:, :4]).float())[0].numpy()
        matched_det = np.zeros(len(detections), dtype=bool)
        matched_gt = np.zeros(len(labels), dtype=bool)
        # 同类且 IoU 达标的 (gt, det) 对，按 IoU 从大到小贪心一对一匹配
        gi_all, di_all = np.nonzero((iou >= self.iou_thres) & (labels[:, 0:1] == detections[:, 5][None, :]))
        if gi_all.size:
            order = np.argsort(-iou[gi_all, di_all])
            for gi, di in zip(gi_all[order], di_all[order]):
                if matched_gt[gi] or matched_det[di]:
                    continue
                matched_gt[gi] = True
                matched_det[di] = True
                self.matrix[int(detections[di, 5]), int(labels[gi, 0])] += 1

        # FP: 未匹配的预测（行=预测类，列=background）
        self.matrix[det_classes[~matched_det], self.nc] += 1
        # FN: 未匹配的 GT（行=background，列=真实类）
        self.matrix[self.nc, labels[~matched_gt, 0].astype(int)] += 1

    def tp_fp(self):
        """Return true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        fn = self.matrix.sum(0) - tp  # false negatives
        return tp[:-1], fp[:-1], fn[:-1]  # remove background class

    def plot(self, normalize=True, save_dir="", names=()):
        """Plot the confusion matrix."""
        try:
            import seaborn as sn
        except ImportError:
            warnings.warn("Seaborn not installed, skipping confusion matrix plot")
            return
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)
        sn.set(font_scale=1.0 if nc < 50 else 0.8)
        labels = (0 < nn < 99) and (nn == nc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                vmax=1.0,
                cbar=False,
            )
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_xticklabels(list(names) + ["background"] if labels else "auto")
        ax.set_yticklabels(list(names) + ["background"] if labels else "auto")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close()
