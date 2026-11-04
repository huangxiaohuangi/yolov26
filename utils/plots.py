# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Plotting utils: Annotator (画框), colors, dataset label stats, training curves."""

import contextlib
import math
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.general import clip_boxes, increment_path, xywh2xyxy, xyxy2xywh

matplotlib.use("Agg")  # 无界面后端，服务器/Windows 都可用


def is_ascii(s=""):
    """Return True if string is composed of only ASCII characters."""
    return all(ord(c) < 128 for c in s)


def check_font(font="Arial.ttf"):
    """Return a font path (fallback to default if not found)."""
    return None


def check_pil_font(font="Arial.ttf", size=10):
    """Return a PIL font (fallback to default)."""
    from PIL import ImageFont

    with contextlib.suppress(Exception):
        return ImageFont.truetype(font, size)
    return ImageFont.load_default()


def colors():
    """Return a color palette (BGR) for class drawing."""
    hexs = (
        "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
        "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
    )
    palette = [hex2rgb(h) for h in hexs]
    return [x[::-1] for x in palette]  # RGB -> BGR


def hex2rgb(h):
    """Convert hex color to RGB tuple."""
    return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class Annotator:
    """Draw boxes / masks / keypoints / text on an image (YOLO 风格).

    Args:
        im (np.ndarray): BGR image.
        line_width (int | None): 线宽，None 时按图片尺寸自适应。
        font_size (int | None): 字体大小。
        font (str): 字体路径。
        pil (bool): 是否使用 PIL 绘制文字（支持中文）。
        example (str): 类别名示例（用于自适应线宽）。
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator."""
        assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images."
        self.im = im if im.flags.writeable else im.copy()
        self.pil = pil
        if self.pil:  # use PIL
            from PIL import Image, ImageDraw

            self.im = Image.fromarray(cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB))
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one bounding box with an optional label."""
        if self.pil:
            from PIL import ImageFont

            p1, p2 = (box[0], box[1]), (box[2], box[3])
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label) if hasattr(self.font, "getsize") else self.font.getbbox(label)[2:]
                outside = p1[1] - h >= 0
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add a rectangle (PIL only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Add text (PIL/cv2)."""
        if self.pil:
            if anchor == "bottom":  # start y at the bottom of the text
                w, h = self.font.getbbox(text)[2:]
                xy[1] -= h
            self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if anchor == "bottom":  # start y at the bottom of the text
                w, h = cv2.getTextSize(text, 0, fontScale=self.lw / 3, thickness=max(self.lw - 1, 1))[0]
                xy[1] -= h + 3
            cv2.putText(
                self.im,
                text,
                (int(xy[0]), int(xy[1])),
                0,
                self.lw / 3,
                txt_color,
                thickness=max(self.lw - 1, 1),
                lineType=cv2.LINE_AA,
            )

    def result(self):
        """Return the annotated image (BGR)."""
        return np.asarray(self.im)[..., ::-1] if self.pil else self.im


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels."""
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007
    return crop


def plot_labels(labels, names=(), save_dir=Path("")):
    """Plot dataset labels: class histogram, box size/center distributions."""
    # Plot labels
    import pandas as pd

    LOGGER = print
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # Scatter matrix
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel("classes")
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 30), cmap="jet", edgecolors="black")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 30), cmap="jet", edgecolors="black")
    ax[2].set_xlabel("width")
    ax[2].set_ylabel("height")
    fig.savefig(Path(save_dir) / "labels.png", dpi=200)
    plt.close(fig)

    # Classes
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax.set_xlabel("classes")
    ax.set_ylabel("count")
    fig.savefig(Path(save_dir) / "labels_class.png", dpi=200)
    plt.close(fig)


def hist2d(x, y, n=100):
    """2d histogram used for plotting labels."""
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_results(file="path/to/results.csv", dir="", on_plot=None, save_dir=None):
    """Plot training results.csv (loss + mAP curves)."""
    import pandas as pd

    save_dir = Path(save_dir) if save_dir else (Path(file).parent if file else Path(dir))
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    keys = [
        "train/box_loss",
        "train/cls_loss",
        "train/l1_loss",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP50",
        "metrics/mAP50-95",
        "val/loss",
        "lr/pg0",
    ]
    fig, ax = plt.subplots(3, 3, figsize=(18, 12), tight_layout=True)
    ax = ax.ravel()
    for f in files:
        try:
            data = pd.read_csv(f)
            x = data[data.columns[0]].values
            for i, k in enumerate(keys):
                if k not in data.columns:
                    continue
                ax[i].plot(x, data[k].values.astype(float), marker=".", linewidth=2, label=f.stem)
                ax[i].set_title(k, fontsize=12)
                ax[i].grid(alpha=0.3)
        except Exception as e:
            print(f"WARNING: Plotting error for {f}: {e}")
    ax[0].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()
