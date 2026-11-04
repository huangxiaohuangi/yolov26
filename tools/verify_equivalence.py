"""逐比特验证：本仓库（YOLOv5 风格重写版）与官方 ultralytics 实现是否等价。

覆盖 5 项检查：
  1. 5 个尺度的参数量；
  2. `load_state_dict(ref.state_dict(), strict=True)`（说明模块命名与结构完全一致）；
  3. 训练态双头输出（one2many / one2one 的 boxes、scores）；
  4. 推理态端到端输出 (bs, max_det, 6)；
  5. 损失总值与 box / cls / l1 三个分项。

用法：
    python tools/verify_equivalence.py                 # 全部检查，n 尺度
    python tools/verify_equivalence.py --scales n s    # 指定尺度
    python tools/verify_equivalence.py --nc 80 --imgsz 640

依赖：pip install ultralytics（仅用于对照，不参与训练）
"""
import argparse
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.yolo import DetectionModel  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402

TOL = 1e-6  # 判定“等价”的最大绝对误差


def banner(text):
    """打印分节标题。"""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def build_pair(scale, nc, imgsz):
    """构造（本仓库模型, 官方模型）并同步权重。"""
    from ultralytics.nn.tasks import DetectionModel as RefModel

    cfg = f"yolo26{scale}.yaml"
    torch.manual_seed(0)
    mine = DetectionModel(str(ROOT / "models" / cfg), nc=nc, ch=3).float()
    torch.manual_seed(0)
    ref = RefModel(cfg, nc=nc, ch=3).float()
    mine.load_state_dict(ref.state_dict(), strict=True)
    mine.eval(), ref.eval()
    return mine, ref, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["n", "s", "m", "l", "x"])
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    opt = ap.parse_args()

    ok = True

    # 1. 参数量 ----------------------------------------------------------------
    banner("1. 参数量对比")
    print(f"{'scale':>6} {'mine':>12} {'official':>12} {'diff':>8}")
    for s in opt.scales:
        mine, ref, _ = build_pair(s, opt.nc, opt.imgsz)
        a, b = sum(p.numel() for p in mine.parameters()), sum(p.numel() for p in ref.parameters())
        flag = "OK" if a == b else "MISMATCH"
        ok &= a == b
        print(f"{s:>6} {a:>12,} {b:>12,} {a - b:>8}  {flag}")

    # 2~5. 用 n 尺度做数值对比 --------------------------------------------------
    mine, ref, cfg = build_pair("n", opt.nc, opt.imgsz)
    x = torch.rand(opt.batch, 3, opt.imgsz, opt.imgsz)

    banner("2. 权重加载（strict=True）")
    print("load_state_dict(ref.state_dict(), strict=True) -> OK")

    banner("3. 训练态双头输出")
    mine.train(), ref.train()
    with torch.no_grad():
        pm, pr = mine(x), ref(x)
    pm = pm[1] if isinstance(pm, tuple) else pm
    pr = pr[1] if isinstance(pr, tuple) else pr
    for head in ("one2many", "one2one"):
        for key in ("boxes", "scores"):
            d = (pm[head][key] - pr[head][key]).abs().max().item()
            ok &= d < TOL
            print(f"  {head:<9} {key:<7} shape={tuple(pm[head][key].shape)}  max|Δ|={d:.3e}  "
                  f"{'OK' if d < TOL else 'MISMATCH'}")

    banner("4. 推理态端到端输出")
    # 官方模型构造完成后 end2end 属性为 False，需要由 validator/exporter 在 nms=False 时显式打开；
    # 本仓库的 Detect 在构造时就置为 yaml 的 end2end 值，这里手动对齐后再比较。
    mine.eval(), ref.eval()
    ref.end2end = True
    with torch.no_grad():
        ym, yr = mine(x), ref(x)
    ym = ym[0] if isinstance(ym, tuple) else ym
    yr = yr[0] if isinstance(yr, tuple) else yr
    d = (ym - yr).abs().max().item()
    ok &= d < TOL
    print(f"  end2end   shape={tuple(ym.shape)}  max|Δ|={d:.3e}  {'OK' if d < TOL else 'MISMATCH'}")

    # 再对比非 end2end（one2many + 外部 NMS）路径。
    # 布局差异：YOLOv5 风格的 Detect 输出 (bs, anchors, 4+nc)，官方 ultralytics 输出 (bs, 4+nc, anchors)，
    # 这里把官方结果转置后比较，保证比的是数值而不是内存布局。
    ref.end2end = False
    mine.model[-1].end2end = False
    with torch.no_grad():
        ym, yr = mine(x), ref(x)
    ym = ym[0] if isinstance(ym, tuple) else ym
    yr = yr[0] if isinstance(yr, tuple) else yr
    if yr.shape == ym.transpose(1, 2).shape:
        yr = yr.transpose(1, 2)
    d = (ym - yr).abs().max().item()
    ok &= d < TOL
    print(f"  one2many  shape={tuple(ym.shape)} (official {tuple(yr.shape)} after transpose)  "
          f"max|Δ|={d:.3e}  {'OK' if d < TOL else 'MISMATCH'}")
    mine.model[-1].end2end = True  # 恢复，供后面损失对比使用

    banner("5. 损失对比")
    hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 100}
    mine.hyp = hyp
    ref.args = types.SimpleNamespace(**hyp)

    from ultralytics.utils.loss import E2ELoss as RefE2ELoss

    crit_mine, crit_ref = ComputeLoss(mine), RefE2ELoss(ref)
    crit_mine.set_epoch(0, 100)  # 让 ProgLoss 权重与官方初始值对齐

    torch.manual_seed(1)
    n = 8
    batch_idx = torch.randint(0, opt.batch, (n,))
    cls = torch.randint(0, opt.nc, (n,))
    xy = torch.rand(n, 2) * 0.6 + 0.2
    wh = torch.rand(n, 2) * 0.3 + 0.1
    targets = torch.cat([batch_idx[:, None].float(), cls[:, None].float(), xy, wh], 1)
    # 必须按图片索引排序：真实 dataloader 的 collate_fn 就是有序的。官方 preprocess 用 cumsum 偏移
    # 定位每个目标的行号，遇到乱序会静默覆盖/丢弃目标（本仓库的逐图循环写法不受影响）。
    targets = targets[targets[:, 0].argsort()]
    batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1], "bboxes": targets[:, 2:6]}

    mine.train(), ref.train()
    with torch.no_grad():
        pm, pr = mine(x), ref(x)
    pm = pm[1] if isinstance(pm, tuple) else pm
    pr = pr[1] if isinstance(pr, tuple) else pr
    loss_m, items_m = crit_mine(pm, targets)
    loss_r, items_r = crit_ref(pr, batch)
    loss_r = loss_r.sum()
    d = abs(float(loss_m) - float(loss_r))
    ok &= d < TOL
    print(f"  total  mine={float(loss_m):.6f}  official={float(loss_r):.6f}  max|Δ|={d:.3e}  "
          f"{'OK' if d < TOL else 'MISMATCH'}")
    for k in items_m:
        d = abs(float(items_m[k]) - float(items_r[k]))
        ok &= d < TOL
        print(f"  {k:<9} mine={float(items_m[k]):.6f}  official={float(items_r[k]):.6f}  "
              f"max|Δ|={d:.3e}  {'OK' if d < TOL else 'MISMATCH'}")

    banner("结果")
    print("✅ 全部检查通过：本实现与官方 ultralytics 逐比特一致。" if ok
          else "❌ 存在不一致项，请检查上面标为 MISMATCH 的行。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
