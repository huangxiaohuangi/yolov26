# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""General utils: logging, yaml, box conversions, NMS, path handling.

本文件保持 YOLOv5 的写法（纯函数 + LOGGER），不依赖 ultralytics 包。
"""

import contextlib
import glob
import inspect
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml

from utils import ROOT, TryExcept, torch_distributed_zero_first

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 Windows 上 OpenMP 重复加载报错

# Settings --------------------------------------------------------------------------------------------------------
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
LOGGING_NAME = "yolo26"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support."""
    level = logging.INFO if verbose and VERBOSE else logging.ERROR
    formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class YoloLogger:
    """自定义 Logger：统一前缀与颜色输出（对应 YOLOv5 的 LOGGER）。"""

    def __init__(self, logger):
        self.logger = logger

    def info(self, msg="", *args, **kwargs):
        self.logger.info(msg % args if args else msg)

    def warning(self, msg="", *args, **kwargs):
        text = colorstr("yellow", "bold", f"WARNING ⚠️  {msg}")
        self.logger.warning(text % args if args else text)

    warn = warning

    def error(self, msg="", *args, **kwargs):
        text = colorstr("red", "bold", f"ERROR ❌  {msg}")
        self.logger.error(text % args if args else text)


LOGGER = YoloLogger(set_logging())


def colorstr(*input):
    """Colors a string using ANSI escape codes, e.g. colorstr('blue', 'bold', 'hello')."""
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


def is_ascii(s=""):
    """Return True if string is composed of only ASCII characters."""
    return all(ord(c) < 128 for c in s)


def check_version(current="0.0.0", minimum="0.0.0", pinned=False, hard=False, verbose=False):
    """Check version against the minimum requirement."""
    current, minimum = (pkg2version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    if hard and not result:
        raise ImportError(f"{minimum} is required, but {current} is currently installed")
    return result


def pkg2version(version):
    """Convert a version string to a comparable tuple."""
    return tuple(int(c) for c in re.findall(r"\d+", str(version))[:3])


def check_imshow(warn=False):
    """Check if environment supports image displays."""
    try:
        if LINUX:
            assert any(k in os.environ for k in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))
            assert Path("/tmp/.X11-unix").exists()
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False
    return True


def check_yaml(file, suffix=(".yaml", ".yml")):
    """Search/download yaml file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix)


def check_file(file, suffix=""):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    if not file or os.path.isdir(file):
        return file
    if Path(file).is_file() or file == "":
        return file
    elif file.startswith(("http:/", "https:/")):
        url = str(Path(file)).replace(":/", "://")  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split("?")[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            LOGGER.info(f"Found {url} locally at {file}")
        else:
            LOGGER.info(f"Downloading {url} to {file}...")
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))
        if not files and not Path(file).exists():
            raise FileNotFoundError(f"'{file}' does not exist")
        return files[0] if len(files) else file


def check_suffix(file="yolo26n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(str(f)).suffix  # file suffix
            if s and s not in suffix:
                raise ValueError(f"{msg}Incorrect suffix for {f}, only {suffix} are supported.")


def check_img_size(imgsz, s=32, floor=0):
    """Verify image size is a multiple of stride s in each dimension, and return the new size."""
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def make_divisible(x, divisor):
    """Return nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_dataset(data, autodownload=True):
    """Download, check and/or unzip dataset if not found locally.

    Args:
        data (dict | str): Dataset yaml dict, or path to the dataset yaml file.
        autodownload (bool): Attempt to download dataset if not found locally.

    Returns:
        (dict): Dataset dict with 'train', 'val', 'nc', 'names' keys and resolved absolute paths.
    """
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(emojis(f"WARNING ⚠️ '{k}' key missing in {data}"))
            data[k] = data.pop("validation")  # replace validation with val
    if "nc" not in data:
        data["nc"] = len(data["names"])
    if isinstance(data["names"], (list, tuple)):  # to dict
        data["names"] = {i: str(x) for i, x in enumerate(data["names"])}

    path = Path(data.get("path") or "")  # optional 'path' default to '.'

    # Download (optional) —— coco128 等小数据集首次运行自动下载
    if data.get("download") and autodownload:
        root = path if str(path) not in ("", ".") else Path("datasets")
        train_p = root / data["train"] if isinstance(data["train"], str) else root / data["train"][0]
        if not train_p.exists():
            import zipfile

            url = data["download"] if isinstance(data["download"], str) else data["download"][0]
            f = root.parent / url2file(url)
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                LOGGER.info(f"Downloading {url} to {f}...")
                torch.hub.download_url_to_file(url, str(f), progress=True)
            with zipfile.ZipFile(f) as z:
                tops = {n.split("/")[0] for n in z.namelist() if n.split("/")[0]}
                if len(tops) == 1 and list(tops)[0] == root.name:
                    z.extractall(root.parent)  # zip 内含同名根目录
                else:
                    root.mkdir(parents=True, exist_ok=True)
                    z.extractall(root)
            LOGGER.info(f"Dataset {root} downloaded and extracted.")

    if str(path) not in ("", "."):
        for k in "train", "val", "test":
            if data.get(k):  # prepend path
                if isinstance(data[k], str):
                    x = (path / data[k]).resolve()
                    if not x.exists() and data[k].startswith("../"):
                        x = (path / data[k][3:]).resolve()
                    data[k] = str(x)
                else:
                    data[k] = [str((path / x).resolve()) for x in data[k]]

    return data


def yaml_load(file="data.yaml", append_filename=False):
    """Load YAML and return as dict, with optional filename appended."""
    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
    if not s.isprintable():  # remove special characters
        s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
    data = yaml.safe_load(s) or {}  # yaml.load(s, Loader=yaml.FullLoader)
    if append_filename:
        data["yaml_file"] = str(file)
    return data


def yaml_save(file="data.yaml", data=None, header=""):
    """Save dict `data` to YAML file."""
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)
    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def print_args(args, show_file=False):
    """Print argparse.Namespace / dict arguments, one key per line (YOLOv5 style)."""
    d = args if isinstance(args, dict) else vars(args)
    s = "".join(f"{colorstr('bold', k)}: {v}\n" for k, v in d.items())
    LOGGER.info(f"\n{colorstr('bright_blue', 'Arguments:')}\n{s}")


def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc."""
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def intersect_dicts(da, db, exclude=()):
    """Return dictionary of intersecting keys with matching shapes, excluding 'exclude' keys."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def strip_optimizer(f="best.pt", s=""):
    """Strip optimizer from 'f' to finalize training, optionally save as 's'."""
    x = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)


def labels_to_class_weights(labels, nc=80):
    """Get class weights (inverse frequency) from training labels."""
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels) - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """Produce image weights based on class_weights and image contents."""
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Return a lambda function for sinusoidal ramp from y1 to y2."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


# Box conversions -------------------------------------------------------------------------------------------------
def xyxy2xywh(x):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, w, h) format."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2)."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalized bounding box coordinates to pixel coordinates."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert bounding box coordinates from (x1, y1, x2, y2) to (x, y, w, h) normalized."""
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def clip_boxes(boxes, shape):
    """Clip bounding boxes to image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale bounding boxes from img1_shape to img0_shape (letterbox aware)."""
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


# NMS -------------------------------------------------------------------------------------------------------------
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_nms=30000,  # maximum number of boxes into torchvision.ops.nms()
    max_wh=7680,  # (pixels) maximum box width and height
    in_place=True,
    rotated=False,
):
    """Runs Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction (torch.Tensor): shape (batch, num_anchors, 4 + nc), already decoded (xyxy + cls scores).
        conf_thres (float): confidence threshold.
        iou_thres (float): IoU threshold for NMS.
        classes (list | None): filter by class, i.e. 0 or [0, 2, 3].
        agnostic (bool): class-agnostic NMS.
        multi_label (bool): each box may have multiple labels.
        labels (tuple): (list of tensors, ...) used for multi_label NMS.
        max_det (int): maximum detections per image.
        nc (int): number of classes, inferred from prediction if 0.
        max_nms (int): maximum number of boxes into torchvision.ops.nms().
        max_wh (int): maximum box width and height.
        in_place (bool): modify input tensors in place.
        rotated (bool): for OBB (not supported here).

    Returns:
        (list[torch.Tensor]): list of detections per image, each of shape (n, 6) = x1, y1, x2, y2, conf, cls.
    """
    if isinstance(prediction, (list, tuple)):  # model in train mode, export mode
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[2] - 4)  # number of classes
    nm = prediction.shape[2] - nc - 4  # number of extra channels (masks / kpts)
    mi = 4 + nc  # mask start index
    xc = prediction[..., 4:mi].amax(2) > conf_thres  # candidates

    # Settings
    if 0 < max_det < 100000:
        pass
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2) if not in_place else prediction.transpose(-1, -2).contiguous()
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output


# Misc ------------------------------------------------------------------------------------------------------------
class Profile(contextlib.ContextDecorator):
    """Profile class for code timing, usable as a decorator or context manager.

    Usage:
        p = Profile()
        with p:
            ...
        print(p.dt)
    """

    def __init__(self, t=0.0, device: torch.device = None):
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_colab():
    """Check if the current script is running inside a Google Colab notebook."""
    return "COLAB_RELEASE_TAG" in os.environ


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(urllib.parse.unquote(urllib.parse.urlparse(str(url)).path)).name


def download_url_to_file(url, file, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path."""
    torch.hub.download_url_to_file(url, file, hash_prefix=hash_prefix, progress=progress)


def get_latest_run(search_dir="."):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


class WorkingDirectory(contextlib.ContextDecorator):
    """Context manager / decorator to temporarily change current working directory."""

    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: B027
        os.chdir(self.cwd)


def methods(instance):
    """Get class methods/attributes as functions for logging."""
    # Get methods and attributes of an object
    for func in [x for x in dir(instance) if callable(getattr(instance, x))]:
        yield func, getattr(instance, func)
