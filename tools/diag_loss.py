"""诊断训练不收敛：检查 dataloader 标签、assigner 前景数、各项 loss 的原始量级。"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\Desktop\yolov26")
sys.path.insert(0, str(ROOT))

from models.yolo import DetectionModel
from utils.dataloaders import create_dataloader
from utils.general import yaml_load
from utils.loss import ComputeLoss

torch.manual_seed(0)
np.random.seed(0)

data = yaml_load(r"D:\Desktop\_smoke\synth.yaml")
hyp = yaml_load(r"D:\Desktop\_smoke\hyp_smoke.yaml")
hyp["epochs"] = 40

train_loader, dataset = create_dataloader(
    str(Path(data["path"]) / data["train"]), 320, 8, 32, hyp=hyp, augment=True, cache=False,
    rect=False, rank=-1, workers=0, prefix="diag: ",
)
print("num imgs", len(dataset), "labels len", len(dataset.labels))
for i in range(3):
    lb = dataset.labels[i]
    print(f"img{i} shape={dataset.shapes[i] if hasattr(dataset,'shapes') else '?'} labels=\n{lb}")

model = DetectionModel(str(ROOT / "models" / "yolo26n.yaml"), nc=data["nc"], ch=3)
model.hyp = hyp
model.nc = data["nc"]
model.names = data["names"]
model = model.float()
model.train()

# --- 复刻 train.py 的 hyp 缩放与 class_weights ---
nl = model.model[-1].nl
hyp["box"] *= 3 / nl
hyp["cls"] *= 3 / nl
from utils.general import labels_to_class_weights
cw = labels_to_class_weights(dataset.labels, data["nc"]) * data["nc"]
print("nl", nl, "hyp box/cls/dfl", hyp["box"], hyp["cls"], hyp.get("dfl"))
print("class_weights", cw)
model.class_weights = cw

crit = ComputeLoss(model)

batch_ = next(iter(train_loader))
imgs, targets = batch_[0], batch_[1]
imgs = imgs.float() / 255.0
print("imgs", imgs.shape, imgs.dtype, "targets", targets.shape, targets.dtype)
print("targets sample\n", targets[:10])
print("targets xywh range", targets[:, 2:6].min().item(), targets[:, 2:6].max().item())

preds = model(imgs)
print("preds type", type(preds))
d = preds[1] if isinstance(preds, tuple) else preds
for k, v in d.items():
    print(" head", k, "feats", [tuple(f.shape) for f in v["feats"]],
          "boxes", tuple(v["boxes"].shape), "scores", tuple(v["scores"].shape))

batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1], "bboxes": targets[:, 2:6]}
for name, branch in [("one2many", crit.criterion.one2many), ("one2one", crit.criterion.one2one)]:
    info, loss, items = branch.get_assigned_targets_and_loss(d[name], batch)
    fg_mask = info[0]
    print(f"\n[{name}] fg anchors = {int(fg_mask.sum())}, loss(raw) = {loss.detach().numpy()}")
    print(f"[{name}] items = { {k: round(float(v), 4) for k, v in items.items()} }")
