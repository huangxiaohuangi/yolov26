# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26 推理脚本（YOLOv5 detect.py 风格）.

Usage:
    $ python predict.py --weights runs/train/exp/weights/best.pt --source data/images/bus.jpg
    $ python predict.py --weights yolo26n.pt --source 0                 # 摄像头
    $ python predict.py --weights yolo26n.pt --source video.mp4 --view-img

输出:
    runs/detect/exp/*.jpg          可视化结果
    runs/detect/exp/labels/*.txt   归一化 xywh 标签（--save-txt）
    runs/detect/exp/crops/...      目标裁剪图（--save-crop）
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO26 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load  # noqa: E402
from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box  # noqa: E402
from utils.torch_utils import select_device, time_sync  # noqa: E402


def batched_nms(det, iou_thres, max_det):
    """Class-wise NMS for end2end detections (bs=1 slice): det (n, 6) -> (k, 6)."""
    if det.shape[0] == 0:
        return det
    keep = torchvision.ops.batched_nms(det[:, :4], det[:, 4], det[:, 5].long(), iou_thres)
    return det[keep[:max_det]]


@torch.no_grad()
def predict_batch(model, im, augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False,
                  max_det=1000, end2end=False):
    """Run inference on a batch and return a list of detections (one (n, 6) tensor per image)."""
    y = model(im)
    if end2end:  # (bs, max_det, 6) xyxy + conf + cls
        det = y[0] if isinstance(y, (tuple, list)) else y
        if augment:  # 水平翻转 TTA，再合并去重
            yf = model(torch.flip(im, [3]))
            df = (yf[0] if isinstance(yf, (tuple, list)) else yf).clone()
            w = im.shape[3]
            x1 = w - df[..., 2]
            x2 = w - df[..., 0]
            df[..., 0], df[..., 2] = x1, x2
            det = torch.cat([det, df], 1)
        out = []
        for d in det:
            d = d[d[:, 4] > conf_thres]
            if classes is not None:
                d = d[(d[:, 5:6] == torch.tensor(classes, device=d.device)).any(1)]
            out.append(batched_nms(d, iou_thres, max_det) if augment else d[:max_det])
        return out

    # 非 end2end：拼接翻转 TTA 的原始输出后统一做 NMS
    if augment:
        yf = model(torch.flip(im, [3]))
        raw = y[0] if isinstance(y, (tuple, list)) else y
        rawf = (yf[0] if isinstance(yf, (tuple, list)) else yf).clone()
        rawf[..., 0] = im.shape[3] - rawf[..., 0]  # xywh 的 x 中心翻转回来
        raw = torch.cat([raw, rawf], 1)
    else:
        raw = y
    return non_max_suppression(raw, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


@torch.no_grad()
def run(weights=ROOT / "yolo26n.pt", source=ROOT / "data/images", imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45,
        max_det=1000, device="", view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False,
        classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=ROOT / "runs/detect",
        name="exp", exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False,
        vid_stride=1, callbacks=Callbacks()):
    """Run inference on images, videos, directories, streams.

    Returns:
        (list): 每张图片的检测结果 (n, 6) [x1, y1, x2, y2, conf, cls]（原图坐标系）。
    """
    callbacks.run("on_predict_start")
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)

    if not (webcam or is_url) and not (Path(source).exists()):
        raise FileNotFoundError(f"Source path '{source}' does not exist")

    save_img = not nosave and not source.endswith(".txt")  # save inference images

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = attempt_load(weights, device=device, inplace=True, fuse=False)
    stride, names = model.stride, model.names
    end2end = getattr(model.model[-1], "end2end", False)
    imgsz = check_img_size(imgsz, s=int(stride.max()))  # check image size
    half &= device.type != "cpu"  # half precision only supported on CUDA
    model.half() if half else model.float()
    LOGGER.info(f"{colorstr('model:')} {weights} ({'end2end/NMS-free' if end2end else 'NMS'})")

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
    dataset = LoadImages(source, img_size=imgsz, stride=int(stride.max()), auto=True, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    results = []
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = predict_batch(model, im, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det, end2end)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    if save_img:
                        annotator.box_label(xyxy, label, color=colors()[c % 20])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Save results
            results.append(det.cpu())
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

            # Stream results
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    callbacks.run("on_predict_end")
    return results


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolo26n.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob, 0 for webcam")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference (flip TTA)")
    parser.add_argument("--visualize", action="store_true", help="visualize features (unused)")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    """Main inference entry."""
    print_args(vars(opt))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
