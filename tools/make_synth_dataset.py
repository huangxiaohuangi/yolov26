"""生成用于冒烟验证的合成检测数据集（两类彩色块），便于快速确认 train.py 能真正收敛。

用法：
    python tools/make_synth_dataset.py --root D:/Desktop/_smoke/dataset2 --train 192 --val 48 --size 256
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

COLORS = {0: (60, 60, 220), 1: (60, 200, 60)}  # BGR: 0=红块, 1=绿块
NAMES = {0: "red_block", 1: "green_block"}


def make_image(rng, size):
    """画一张带 1~3 个彩色块的图，返回 (img, labels)。labels 为 normalized xywh。"""
    img = np.full((size, size, 3), rng.integers(90, 150), dtype=np.uint8)
    img = cv2.add(img, rng.integers(-25, 25, img.shape, dtype=np.int16).clip(-255, 255).astype(np.uint8))

    labels = []
    n = rng.integers(1, 4)
    for _ in range(n):
        cls = int(rng.integers(0, 2))
        w = int(rng.integers(size // 5, size // 2))
        h = int(rng.integers(size // 5, size // 2))
        x1 = int(rng.integers(0, size - w))
        y1 = int(rng.integers(0, size - h))
        color = COLORS[cls]
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
        # 加一点内部噪声，避免颜色完全一致导致“太容易”
        patch = img[y1:y1 + h, x1:x1 + w].astype(np.int16)
        patch += rng.integers(-18, 18, patch.shape, dtype=np.int16)
        img[y1:y1 + h, x1:x1 + w] = patch.clip(0, 255).astype(np.uint8)
        labels.append([cls, (x1 + w / 2) / size, (y1 + h / 2) / size, w / size, h / size])
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/Desktop/_smoke/dataset2")
    ap.add_argument("--train", type=int, default=192)
    ap.add_argument("--val", type=int, default=48)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    opt = ap.parse_args()

    root = Path(opt.root)
    rng = np.random.default_rng(opt.seed)
    for split, n in (("train", opt.train), ("val", opt.val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = make_image(rng, opt.size)
            cv2.imwrite(str(img_dir / f"{split}_{i:04d}.jpg"), img)
            with open(lbl_dir / f"{split}_{i:04d}.txt", "w") as f:
                for lb in labels:
                    f.write(f"{lb[0]} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")
        print(f"{split}: {n} images -> {img_dir}")

    yaml_path = root / "synth2.yaml"
    yaml_path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnc: 2\n"
        f"names: {{0: red_block, 1: green_block}}\n",
        encoding="utf-8",
    )
    print("data yaml ->", yaml_path)


if __name__ == "__main__":
    main()
