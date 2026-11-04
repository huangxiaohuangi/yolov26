# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26 验证/测试脚本（YOLOv5 风格）.

Usage:
    $ python val.py --weights runs/train/exp/weights/best.pt --data data/coco128.yaml --img 640
    $ python val.py --data data/coco128.yaml --weights '' --cfg yolo26n.yaml   # 随机权重自检
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO26 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER,
    check_dataset,
    check_img_size,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

VAL_HYP = {  # 验证时关闭所有增强
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0,
    "hsv_v": 0.0, "cutout": 0.0,
}


def save_one_txt(predn, save_conf, shape, file):
    """Save one txt prediction file (normalized xywh)."""
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def process_batch(detections, labels, iouv):
    """Return correct prediction matrix (N, 10) for one image.

    Args:
        detections (np.ndarray): (N, 6) [x1, y1, x2, y2, conf, cls].
        labels (np.ndarray): (M, 5) [cls, x1, y1, x2, y2].
        iouv (np.ndarray): IoU thresholds.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(torch.from_numpy(labels[:, 1:]).float(), torch.from_numpy(detections[:, :4]).float())[0].numpy()
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [det_idx, gt_idx, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


class Profile:
    """极简计时器."""

    def __init__(self):
        self.t = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.dt = time.time() - self.start
        self.t += self.dt


@torch.no_grad()
def run(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task="val",
        device="", workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False,
        save_conf=False, save_json=False, project=ROOT / "runs/val", name="exp", exist_ok=False, half=True,
        dnn=False, plots=True, callbacks=Callbacks(), compute_loss=None, model=None, dataloader=None, save_dir=None):
    """Run validation on a dataset.

    Args:
        data (str | dict): dataset yaml.
        weights (str | list | None): 权重路径；若已传入 model 可省略。
        model (nn.Module | None): 直接传入模型（train.py 内部调用）。
        compute_loss (ComputeLoss | None): 若提供则同时统计 val loss。
        dataloader: 复用外部 dataloader（train.py 内部调用）。

    Returns:
        (tuple): (mp, mr, map50, map, val_loss)
    """
    if model is None:
        from models.experimental import attempt_load

        model = attempt_load(weights, device=select_device(device), inplace=True, fuse=False)

    device = next(model.parameters()).device
    half &= device.type != "cpu"  # half precision only supported on CUDA
    model.half() if half else model.float()
    imgsz = int(imgsz[0]) if isinstance(imgsz, (list, tuple)) else int(imgsz)
    imgsz = check_img_size(imgsz, s=int(model.stride.max()))

    # Configure
    model.eval()
    nc = int(getattr(model, "nc", 0) or model.yaml.get("nc", 80))  # number of classes（EMA 副本可能还没同步 nc）
    names = {k: v for k, v in enumerate(model.names)} if hasattr(model, "names") else {i: str(i) for i in range(nc)}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    end2end = getattr(model.model[-1], "end2end", False)  # YOLO26 NMS-free 推理

    # Dataloader
    if dataloader is None:
        data = check_dataset(data)
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            int(model.stride.max()),
            single_cls=single_cls,
            hyp=VAL_HYP,
            rect=True,  # rect 验证
            rank=-1,
            workers=workers,
            pad=0.5,
            prefix=colorstr(f"{task}: "),
        )[0]
    else:
        data = check_dataset(data)

    if save_txt:
        save_dir = Path(save_dir or increment_path(Path(project) / name, exist_ok=exist_ok))
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc) if plots else None
    loss_items = None
    stats = []
    dt = (Profile(), Profile(), Profile())
    t0 = time.time()

    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        with dt[0]:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape

        # Inference
        with dt[1]:
            preds = model(im)
            if end2end:  # (bs, max_det, 6) 已是 NMS-free 输出
                out = preds[0] if isinstance(preds, (tuple, list)) else preds
                out = [x[x[:, 4] > conf_thres][:max_det] for x in out]
            else:
                out = non_max_suppression(preds, conf_thres, iou_thres, multi_label=True, max_det=max_det)

        # Loss（注意：损失用的是归一化标签，必须在下面转像素之前计算）
        if compute_loss is not None:
            with dt[2]:
                _, items = compute_loss(preds, targets)
                loss_items = items if loss_items is None else {
                    k: loss_items[k] + items[k] for k in items
                }

        # 标签归一化 xywh -> letterbox 图像素 xywh（YOLOv5 的关键一步，否则与预测框不在同一坐标系）
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if confusion_matrix is not None:
                        confusion_matrix.process_batch(detections=None, labels=labels.cpu().numpy())
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn.cpu().numpy(), labelsn.cpu().numpy(), iouv.cpu().numpy()).to(device)
                if confusion_matrix is not None:
                    confusion_matrix.process_batch(predn.cpu().numpy(), labelsn.cpu().numpy())

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].cpu()))

            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")

    # Compute metrics
    if stats:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
    else:
        stats = [np.zeros((0, niou), dtype=bool), np.zeros(0), np.zeros(0), np.zeros(0)]

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, _ = ap_per_class(
            *stats, plot=plots, save_dir=save_dir or Path("."), names=names
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0
        ap_class = []

    # number of targets per class（即使没有正确预测也要统计）
    nt = np.bincount(stats[3].astype(int), minlength=nc) if len(stats[3]) else np.zeros(nc)

    # Print results
    pf = "%20s" + "%12.3g" * 6  # print format (class, images, instances, P, R, mAP50, mAP50-95)
    LOGGER.info(f"{'Class':>20s}{'Images':>12s}{'Instances':>12s}{'P':>10s}{'R':>10s}{'mAP50':>10s}{'mAP50-95':>10s}")
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nc < 50 and len(ap_class):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x.t / max(seen, 1) * 1e3 for x in dt)  # speeds per image
    LOGGER.info("Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape %s"
                % (*t, (1, 3, imgsz, imgsz)))

    if confusion_matrix is not None:
        confusion_matrix.plot(save_dir=save_dir or Path("."), names=list(names.values()))

    model.float()  # for training
    val_loss = float(sum(loss_items.values())) if loss_items else 0.0
    return mp, mr, map50, map, val_loss


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, default=ROOT / "yolo26n.pt", help="model path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path (when --weights is empty)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="val or test")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--plots", action="store_true", default=True, help="save plots")
    return parser.parse_args()


def main(opt):
    """Main validation entry."""
    print_args(vars(opt))
    if opt.weights:
        run(**{k: v for k, v in vars(opt).items() if k != "cfg"})
    else:
        from models.yolo import Model

        device = select_device(opt.device)
        model = Model(opt.cfg, nc=None).to(device)
        model.hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        model.nc = model.yaml["nc"]
        model.names = [str(i) for i in range(model.nc)]
        run(**{k: v for k, v in vars(opt).items() if k != "cfg"}, model=model)


if __name__ == "__main__":
    main(parse_opt())
