"""固定单 batch 过拟合测试：确认 loss 能否稳定下降（排除优化器/梯度断流问题）。"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\Desktop\yolov26")
sys.path.insert(0, str(ROOT))

from models.yolo import DetectionModel
from utils.dataloaders import create_dataloader
from utils.general import labels_to_class_weights, yaml_load
from utils.loss import ComputeLoss
from utils.torch_utils import smart_optimizer

OPT = sys.argv[1] if len(sys.argv) > 1 else "MuSGD"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

torch.manual_seed(0)
np.random.seed(0)

data = yaml_load(r"D:\Desktop\_smoke\synth.yaml")
hyp = yaml_load(r"D:\Desktop\_smoke\hyp_smoke.yaml")
hyp["epochs"] = 40

loader, dataset = create_dataloader(
    str(Path(data["path"]) / data["train"]), 320, 8, 32, hyp=hyp, augment=False,
    cache=False, rect=False, rank=-1, workers=0, prefix="",
)
imgs, targets = next(iter(loader))[0:2]
imgs = imgs.float() / 255

model = DetectionModel(str(ROOT / "models" / "yolo26n.yaml"), nc=data["nc"], ch=3)
model = model.float().train()
nl = model.model[-1].nl
hyp["box"] *= 3 / nl
hyp["cls"] *= 3 / nl
hyp["dfl"] = hyp.get("dfl", 1.5)
model.hyp = hyp
model.class_weights = labels_to_class_weights(dataset.labels, data["nc"]) * data["nc"]

opt = smart_optimizer(model, OPT, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
crit = ComputeLoss(model)
crit.set_epoch(0, 40)

print(f"optimizer={OPT} steps={STEPS} n_targets={targets.shape[0]}")
for i in range(STEPS):
    pred = model(imgs)
    loss, items = crit(pred, targets)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    if i % 25 == 0 or i == STEPS - 1:
        d = {k: round(float(v), 4) for k, v in items.items()}
        print(f"step {i:4d}  total={float(loss):9.4f}  {d}")
