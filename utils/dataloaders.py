# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dataloaders and dataset utils (YOLOv5 风格).

标签格式与 YOLOv5 完全一致：每张图一个 .txt，每行 `class x_center y_center width height`（归一化到 0-1）。
"""

import glob
import hashlib
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

from utils.augmentations import (
    augment_hsv,
    copy_paste,
    cutout,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    colorstr,
    xywhn2xyxy,
    xyxy2xywhn,
)

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def img2label_paths_(img_paths):
    """Define label paths as a function of image paths (see utils.general.img2label_paths)."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def verify_image(args):
    """Verify one image."""
    (f, nf, nm, ne, nc, i) = args
    # `nf` files scanned, `nm` missing, `ne` empty, `nc` corrupted
    try:
        # verify images
        im = Image.open(f)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"

        # verify labels
        if os.path.isfile(nf):
            with open(nf, errors="ignore") as t:
                lb = np.array([x.split() for x in t.read().strip().splitlines()], dtype=np.float32)  # labels
            assert len(lb), f"empty label file {nf}"
            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
            assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
            assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
            _, i = np.unique(lb, axis=0, return_index=True)
            if len(i) < lb.shape[0]:
                lb = lb[i]
    except Exception as e:
        if ne == 0:
            LOGGER.warning(f"{colorstr('red', 'WARNING')} {f}: {e}")
        nf = 0  # remove
    return f, nf, shape


def verify_image_label(args):
    """Verify one image-label pair, return (im_file, lb, shape, segments, nm, nf, ne, nc, msg)."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, im_file, 0, 0, "", []  # number (missing, found, empty, corrupt), message
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, errors="ignore") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


class LoadImagesAndLabels(Dataset):
    """YOLO26/YOLOv5 训练 & 验证数据集.

    Args:
        path (str): 图片目录、txt 列表或 glob。
        img_size (int): 训练尺寸。
        batch_size (int): batch size（rect 模式用于分桶）。
        augment (bool): 是否启用数据增强。
        hyp (dict): 超参数字典（mosaic/mixup/degrees...）。
        rect (bool): rectangular training/val。
        cache_images (bool | str): True / 'ram' / 'disk'。
        single_cls (bool): 把所有类别合并为 1 类。
        stride (int): 最大 stride。
        pad (float): rect 验证时的 padding 比例。
        prefix (str): 日志前缀。
    """

    cache_version = "1.0"  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=""):
        """Initialize the dataset."""
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = None

        self.img_files = []
        p = Path(path)
        if p.is_dir():
            self.img_files = sorted(glob.glob(str(p / "**" / "*.*"), recursive=True))
            self.img_files = [x for x in self.img_files if x.split(".")[-1].lower() in IMG_FORMATS]
        elif p.is_file() and p.suffix == ".txt":
            with open(p, errors="ignore") as f:
                self.img_files = sorted(
                    x.strip().replace("./", "") for x in f.read().splitlines() if len(x.strip())
                )
        else:  # glob
            self.img_files = sorted(glob.glob(str(path)))
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {path}")
        assert self.img_files, f"{prefix}No images found"
        self.im_files = self.img_files  # 后续可能被 cache 更新（过滤损坏图片）

        # Check cache
        self.label_files = img2label_paths_(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.img_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            LOGGER.info(f"{prefix}{d}")
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths_(self.im_files)
        n = len(shapes)  # number of images

        # Filter
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
            if single_cls:  # single-class training, for single-class datasets (i.e. 'train-single-cls')
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = Tqdm(results, total=n, bar_format=TQDM_BAR_FORMAT, disable=os.getenv("LOCAL_RANK", -1) != -1)
            for i, x in enumerate(pbar):
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized
                    b += self.ims[i].nbytes
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Cache dataset labels, check images and read shapes."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix)))
            pbar = Tqdm(results, total=len(self.im_files), bar_format=TQDM_BAR_FORMAT, desc=desc)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. See dataset docs.")
        x["hash"] = get_hash(self.label_files + self.img_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")
        return x

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.im_files)

    def __getitem__(self, index):
        """Returns one image and its labels, with all training augmentations applied."""
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, _ = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            # Albumentations (optional, 未启用)
            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Cutout
            labels = cutout(img, labels, p=hyp.get("cutout", 0.0)) if nl else labels

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.img_size / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            elif (h0, w0) != (self.img_size, self.img_size):  # square resize
                im = cv2.resize(im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def load_mosaic(self, index):
        """Loads a 4-mosaic image: 4 images combined into one with random center."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [x * np.array([w, h, w, h]) + np.array([padw, padh, padw, padh]) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4, _ = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class Tqdm:
    """Minimal tqdm wrapper: 若未安装 tqdm 则退化为普通迭代器。"""

    def __init__(self, iterable=None, total=None, desc="", bar_format=None, disable=False):
        try:
            from tqdm import tqdm

            self.pbar = tqdm(iterable, total=total, desc=desc, bar_format=bar_format, disable=disable)
            self.iterable = None
        except ImportError:
            self.pbar = None
            self.iterable = iterable
            self.total = total
            self.desc = desc

    def __iter__(self):
        return iter(self.pbar if self.pbar is not None else self.iterable)

    def __setattr__(self, key, value):
        if key in ("desc", "n") and hasattr(self, "pbar") and self.pbar is not None:
            setattr(self.pbar, key, value)
        else:
            super().__setattr__(key, value)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False,
                      pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix=""):
    """Creates train or validation dataloader (YOLOv5 signature).

    Args:
        path (str): 图片目录 / txt / glob。
        imgsz (int): 图片尺寸。
        batch_size (int): batch size。
        stride (int): 最大 stride。
        single_cls (bool): 单类训练。
        hyp (dict): 超参数。
        augment (bool): 是否增强（训练 True）。
        cache (bool | str): 缓存方式。
        pad (float): rect 模式 padding。
        rect (bool): rectangular。
        rank (int): DDP rank（-1 表示单进程）。
        workers (int): dataloader workers。
        image_weights (bool): 按类别平衡采样。
        quad (bool): 兼容 YOLOv5 的参数（未实现 4x 拼接，忽略）。
        prefix (str): 日志前缀。

    Returns:
        (tuple): dataloader, dataset。
    """
    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augment images
        hyp=hyp,  # augmentation hyperparameters
        rect=rect,  # rectangular training
        cache_images=cache,
        single_cls=single_cls,
        stride=int(stride),
        pad=pad,
        image_weights=image_weights,
        prefix=prefix,
    )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=augment)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + rank)
    return (
        loader(
            dataset,
            batch_size=batch_size,
            shuffle=augment and rank == -1,
            num_workers=nw,
            sampler=sampler,
            pin_memory=PIN_MEMORY,
            collate_fn=LoadImagesAndLabels.collate_fn,
            worker_init_fn=seed_worker_,
            generator=generator,
        ),
        dataset,
    )


def seed_worker_(worker_id):
    """Set dataloader worker seed (reproducible augmentation)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _RepeatSampler:
    """Sampler that repeats forever (for InfiniteDataLoader)."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers; object attributes are shared between iterations."""

    def __init__(self, *args, **kwargs):
        """Initializes the infinite dataloader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class LoadImages:
    """YOLO26 inference dataloader for images/videos/directories/streams.

    Yields:
        (tuple): (path, img, im0s, vid_cap, s).
    """

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initialize LoadImages."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f"No images or videos found in {p}. Supported formats are: {IMG_FORMATS} and {VID_FORMATS}"

    def __iter__(self):
        """Returns an iterator of images/videos."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next image/video frame."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "
        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f"Image Not Found {path}")
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Create a new video capture object."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the length of the dataset."""
        return self.nf  # number of files
