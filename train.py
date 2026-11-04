# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO26 on a custom dataset (YOLOv5 风格训练脚本).

Usage - 单卡 / CPU 训练:
    $ python train.py --data data/coco128.yaml --cfg yolo26n.yaml --img 640 --batch 16 --epochs 100
    $ python train.py --data data/coco128.yaml --weights runs/train/exp/weights/last.pt --resume

与 YOLOv5 的差异（重点）：
    1. 损失：anchor-free + TAL，无 obj 分支，box 回归用 CIoU + 归一化 L1（YOLO26 去掉 DFL）；
    2. 双头训练：one2many(topk=10) + one2one(topk=7->1)，权重按 ProgLoss 从 0.8 衰减到 0.1；
    3. 优化器：默认 MuSGD（Muon + SGD 混合），也可选 SGD/Adam/AdamW；
    4. 不再需要 AutoAnchor（anchor-free），但保留了 YOLOv5 的训练流程（warmup/EMA/多尺度/早停）。
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO26 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# DDP --------------------------------------------------------------------------------------------------------------
# torchrun 会注入 LOCAL_RANK / RANK / WORLD_SIZE；旧的 `python -m torch.distributed.launch` 只传 --local_rank 参数，
# 所以在 main() 里还会用 opt.local_rank 兜底。
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

import val as validate  # for end-of-epoch mAP  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import Tqdm, create_dataloader  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER,
    TORCH_VERSION,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
    print_args,
    strip_optimizer,
    torch_load,
    yaml_load,
    yaml_save,
)
from utils.loss import ComputeLoss  # noqa: E402
from utils.metrics import fitness  # noqa: E402
from utils.torch_utils import (  # noqa: E402
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_optimizer,
)


def train(hyp, opt, device, callbacks):
    """Train a YOLO26 model.

    Args:
        hyp (str | dict): 超参数 yaml 路径或 dict。
        opt (argparse.Namespace): 命令行参数。
        device (torch.device): 训练设备。
        callbacks (Callbacks): 回调。

    Returns:
        (tuple): 最终验证结果 (P, R, mAP50, mAP50-95, val_loss)。
    """
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")
    if RANK not in {-1, 0}:  # 非主进程只输出错误，避免多进程刷屏
        LOGGER.logger.setLevel(logging.ERROR)

    # Directories
    w = save_dir / "weights"  # weights dir
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        hyp = yaml_load(hyp)  # load hyps dict
    hyp = {k: (v if not isinstance(v, str) else v) for k, v in hyp.items()}
    if RANK in {-1, 0}:
        LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings（只让主进程写，避免多进程同时写同一文件）
    if RANK in {-1, 0}:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Config
    cuda = device.type != "cpu"
    init_seeds(opt.seed + max(RANK, 0), deterministic=True)  # DDP 下各 rank 用不同种子
    data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    # Model
    pretrained = str(weights).endswith(".pt") and Path(weights).is_file()
    if pretrained:
        ckpt = torch_load(weights, map_location="cpu")  # load checkpoint to CPU
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        if str(weights).endswith(".pt"):
            LOGGER.warning(f"WARNING ⚠️ weights {weights} not found, training from scratch")
        model = Model(cfg or "yolo26n.yaml", ch=3, nc=nc).to(device)  # create

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size（DDP 下 --batch-size 是「单卡」batch，全局 batch = batch_size * WORLD_SIZE）
    if batch_size == -1:  # 自动 batch：按显存粗略估计（CPU 固定 16）
        batch_size = 16 if not cuda else max(1, int(torch.cuda.get_device_properties(0).total_memory / 1e9 / 2))
    batch_size = max(int(batch_size), 1)

    # SyncBatchNorm（必须在 DDP 包装之前）
    if getattr(opt, "sync_bn", False) and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        if RANK == 0:
            LOGGER.info("Using SyncBatchNorm()")

    # DDP 包装
    if RANK != -1:
        if cuda:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, broadcast_buffers=False
            )
        else:  # CPU DDP（gloo）
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)

    # Optimizer
    nbs = 64  # nominal batch size（全局，即 batch_size * 卡数）
    global_batch = batch_size * max(WORLD_SIZE, 1)
    accumulate = max(round(nbs / global_batch), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= global_batch * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"],
                                muon_scope=getattr(opt, "muon_scope", "auto"))

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:

        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            ckpt = torch_load(weights, map_location="cpu")
            csd = ckpt["model"].float().state_dict()
            model.load_state_dict(csd, strict=False)
            if ckpt.get("ema") is not None:
                ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
                ema.updates = ckpt.get("updates", 0)
            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt.get("best_fitness", 0.0) or 0.0
            start_epoch = ckpt["epoch"] + 1
            if epochs < start_epoch:
                LOGGER.info(f"{colorstr('resume:')} epochs={epochs} < start_epoch={start_epoch}, "
                            f"setting epochs={start_epoch + 100}")
                epochs = start_epoch + 100
            LOGGER.info(f"{colorstr('resume:')} resuming from epoch {start_epoch} (best_fitness={best_fitness:.4f})")
        del ckpt

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=RANK,
        workers=workers,
        image_weights=opt.image_weights,
        prefix=colorstr("train: "),
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Val loader（注意：rank 必须是 -1。验证只在主进程跑，若分片会只剩 1/WORLD_SIZE 的验证集）
    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size * 2,
        gs,
        single_cls,
        hyp=hyp,
        cache=None if noval else opt.cache,
        rect=True,
        rank=-1,
        workers=workers * 2,
        pad=0.5,
        prefix=colorstr("val: "),
    )[0]

    callbacks.run("on_pretrain_routine_end", labels, names)

    # Model attributes（DDP 包装后必须设在内部模型上，否则 ComputeLoss / ModelEMA 读不到）
    core = de_parallel(model)
    nl = core.model[-1].nl  # number of detection layers
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= 3 / nl  # scale to layers（注意：YOLO26 官方损失权重不随类别数缩放，这里也不做 nc/80）
    hyp["dfl"] = hyp.get("dfl", 1.5)
    hyp["label_smoothing"] = opt.label_smoothing
    hyp["epochs"] = epochs
    core.nc = nc  # attach number of classes to model
    core.hyp = hyp  # attach hyperparameters to model
    core.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    core.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    if accumulate > nb:  # 小数据集保护：保证每个 epoch 至少真正优化一次
        LOGGER.info(f"accumulate {accumulate} -> {max(1, nb)} (nb={nb} batches/epoch, batch_size={batch_size})")
        accumulate = max(1, nb)
    nw = max(round(hyp["warmup_epochs"] * nb), 100) if hyp["warmup_epochs"] > 0 else 0  # warmup iterations
    last_opt_step = -1
    first_opt_step = True  # 首次 optimizer.step() 打耗时，便于定位「卡住」
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP50, mAP50-95, val_loss
    scheduler.last_epoch = start_epoch - 1  # do not move
    amp = bool(cuda and opt.amp)  # AMP 仅在 CUDA 上启用
    if TORCH_VERSION >= (2, 4):  # torch.amp 命名空间（2.4+）
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=amp)  # noqa: E731
    else:  # torch 1.7 ~ 2.3 只有 torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=amp)  # noqa: E731
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    fresults = None
    if RANK in {-1, 0}:  # 只有主进程写 results.csv
        fresults = open(save_dir / "results.csv", "w")
        fresults.write("epoch,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision,metrics/recall,"
                       "metrics/mAP50,metrics/mAP50-95,val/loss,"
                       + ",".join(f"lr/pg{i}" for i in range(len(optimizer.param_groups))) + "\n")

    for epoch in range(start_epoch, epochs):  # epoch -------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()
        compute_loss.set_epoch(epoch, epochs)  # ProgLoss 权重更新
        if RANK != -1:  # DDP：更新 DistributedSampler 并重建 iterator，保证每 epoch 重新打乱
            train_loader.set_epoch(epoch)

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "l1_loss", "Instances", "Size"))
        try:  # 非主进程不显示进度条；旧版 Tqdm 没有 disable 参数时降级
            pbar = Tqdm(pbar, total=nb, disable=RANK not in {-1, 0})
        except TypeError:
            pbar = Tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch ----------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, min(np.interp(ni, xi, [1, nbs / global_batch]).round(), nb))
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with autocast():
                pred = model(imgs)  # forward (train mode -> dict)
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                t_opt = time.time()
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
                if first_opt_step:  # 首次 step 打耗时：MuSGD 在 GPU 上对大卷积核做 Newton-Schulz 会很慢
                    first_opt_step = False
                    dt_opt = time.time() - t_opt
                    msg = f"首次 optimizer.step() 耗时 {dt_opt:.2f}s"
                    if dt_opt > 3.0 and opt.optimizer == "MuSGD":
                        msg += (f"（muon_scope={getattr(opt, 'muon_scope', 'auto')}，"
                                f"可加 --muon-scope linear 或 --optimizer SGD 显著加速）")
                    LOGGER.info(msg)

            # Log
            li = torch.stack([loss_items["box_loss"], loss_items["cls_loss"],
                              loss_items.get("l1_loss", loss_items.get("dfl_loss", torch.zeros((), device=device)))]
                             ).to(device).float()
            mloss = (mloss * i + li) / (i + 1)
            mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            pbar.desc = ("%11s" * 2 + "%11.4g" * 5) % (
                f"{epoch}/{epochs - 1}", mem, *mloss.tolist(), targets.shape[0], imgs.shape[-1]
            )
            callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss.tolist()))
            if callbacks.stop_training:
                break
            # end batch ---------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        ema.update_attr(de_parallel(model), include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])

        if RANK in {-1, 0}:  # 只有主进程做验证 / 保存 / 写日志
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results = validate.run(
                    data_dict,
                    batch_size=batch_size * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )
                maps = np.zeros(nc)

            # Update best mAP
            fi = fitness(np.array(results[:4]).reshape(1, -1))  # weighted combination of [P, R, mAP50, mAP50-95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            best_fitness = max(best_fitness, fi)
            callbacks.run("on_fit_epoch_end", list(mloss.tolist()) + list(results), epoch, best_fitness, fi)

            # Save model
            if not nosave or (final_epoch and not opt.noval):
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "names": names,
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

            # Write results.csv
            fresults.write(("%d," % epoch)
                           + ",".join(f"{x:.6g}" for x in mloss.tolist())
                           + "," + ",".join(f"{x:.6g}" for x in results[:4])
                           + f",{results[4]:.6g}"
                           + "," + ",".join(f"{x:.6g}" for x in lr) + "\n")
            fresults.flush()

        # 同步早停标志：否则 rank0 提前退出、其余 rank 继续跑，DDP 集合通信会死锁
        if RANK != -1:
            stop_t = torch.tensor(int(stop), device=device)
            torch.distributed.all_reduce(stop_t, op=torch.distributed.ReduceOp.MAX)
            stop = bool(stop_t.item())

        if stop:
            break  # must break all DDP ranks
        # end epoch ------------------------------------------------------------------------------------------------
    # end training -------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:  # 收尾工作（strip / 最终验证 / 画曲线）只在主进程做
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results = validate.run(
                        data_dict,
                        batch_size=batch_size * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )

        callbacks.run("on_train_end", last, best, epoch, results)
        fresults.close()

        # 训练曲线
        try:
            from utils.plots import plot_results

            plot_results(save_dir=save_dir)
        except Exception as e:
            LOGGER.warning(f"WARNING ⚠️ plot results failed: {e}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """Parse command-line arguments for YOLO26 training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="initial weights path (*.pt), '' for scratch")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path, i.e. yolo26n.yaml")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size, -1 for auto")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["MuSGD", "SGD", "Adam", "AdamW", "NAdam"], default="MuSGD",
                        help="optimizer (YOLO26 default: MuSGD); NAdam needs torch>=1.10, else falls back to Adam")
    parser.add_argument("--muon-scope", type=str, choices=["auto", "all", "linear", "none"], default="auto",
                        help="MuSGD 的 Muon 作用范围：auto(GPU=all/CPU=linear) | all(所有 2D/4D 权重，"
                             "yolo26x 单卡会很慢) | linear(只 2D 权重和 1x1 卷积) | none(退化为纯 SGD)")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm (DDP only, not implemented)")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader (ignored)")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--amp", action="store_true", help="use Automatic Mixed Precision (CUDA only)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP argument, do not modify")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=None):
    """Entry point: prepare directories and start training (CPU / 单卡 / DDP)."""
    global LOCAL_RANK, RANK, WORLD_SIZE

    # 兼容旧的 `python -m torch.distributed.launch`：它只传 --local_rank 参数，不设环境变量
    if getattr(opt, "local_rank", -1) != -1 and LOCAL_RANK == -1:
        LOCAL_RANK = opt.local_rank
        RANK = int(os.getenv("RANK", opt.local_rank))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

    if callbacks is None:
        callbacks = Callbacks()
    if RANK in {-1, 0}:  # 只在主进程打印参数
        print_args(vars(opt))

    # Resume (from specified or most recent last.pt)
    if opt.resume and not opt.weights:
        ckpt_path = Path(opt.resume) if isinstance(opt.resume, str) and Path(str(opt.resume)).is_file() else None
        if ckpt_path is None:
            last = sorted(Path(ROOT / "runs").rglob("last.pt"), key=os.path.getmtime)
            assert last, "No last.pt found to resume from"
            ckpt_path = last[-1]
        ckpt = torch_load(ckpt_path, map_location="cpu")
        opt.weights = str(ckpt_path)
        opt.cfg = ckpt.get("model").yaml if ckpt.get("model") is not None else opt.cfg
        if isinstance(ckpt.get("opt"), dict):
            saved = ckpt["opt"]
            for k, v in saved.items():
                if k in ("weights", "cfg", "resume", "save_dir", "hyp", "local_rank"):
                    continue
                setattr(opt, k, v)
        LOGGER.info(f"Resuming from {ckpt_path}")

    # Checks
    assert opt.cfg or opt.weights, "Either --cfg or --weights must be specified"
    if opt.cfg:
        opt.cfg = check_yaml(opt.cfg)
    if opt.data:
        opt.data = check_yaml(opt.data)
    if opt.hyp:
        opt.hyp = check_yaml(opt.hyp)

    device = select_device(opt.device, batch=opt.batch_size)

    # DDP：每个进程绑定自己的卡（select_device 返回的 cuda:0 在 DDP 下必须覆盖）
    if LOCAL_RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, (
            f"--batch-size {opt.batch_size} 必须是 WORLD_SIZE={WORLD_SIZE} 的整数倍"
        )
        if torch.cuda.is_available():
            assert torch.cuda.device_count() > LOCAL_RANK, (
                f"CUDA 卡数 ({torch.cuda.device_count()}) 少于 LOCAL_RANK={LOCAL_RANK}"
            )
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device("cuda", LOCAL_RANK)
            backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        else:  # 无 GPU 时走 gloo（可用于逻辑调试，性能无意义）
            device = torch.device("cpu")
            backend = "gloo"
        # 默认 env://（torchrun 会设 MASTER_ADDR/PORT）。某些 torch 构建不带 libuv，
        # TCPStore 会报 "use_libuv was requested..."，此时可用 YOLO_DDP_INIT=file:///tmp/ddp_init 绕过。
        init_method = os.getenv("YOLO_DDP_INIT", "env://")
        torch.distributed.init_process_group(
            backend=backend, init_method=init_method, timeout=timedelta(seconds=5400)
        )
        if RANK == 0:
            LOGGER.info(f"{colorstr('DDP:')} world_size={WORLD_SIZE}, device={device}, backend={backend}")
    elif opt.device and "," in str(opt.device):
        LOGGER.warning(
            f"device={opt.device} 指定了多张卡，但没有检测到 DDP 环境变量（LOCAL_RANK），实际只会用 cuda:0。"
            f"多卡请用：torchrun --nproc_per_node={len(str(opt.device).split(','))} train.py ..."
        )

    # Save directory（DDP 下必须让所有 rank 用同一个目录，否则各进程 increment_path 会得到不同序号）
    save_path = Path(opt.project) / opt.name
    if LOCAL_RANK != -1:
        marker = save_path.parent / f".{opt.name}.ddp_dir"
        if RANK == 0:
            save_path = increment_path(save_path, exist_ok=opt.exist_ok)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(save_path), encoding="utf-8")
        torch.distributed.barrier()  # 等 rank0 写完
        save_path = Path(marker.read_text(encoding="utf-8").strip())
        torch.distributed.barrier()  # 等所有 rank 读完再删
        if RANK == 0 and marker.exists():
            marker.unlink()
    else:
        save_path = increment_path(save_path, exist_ok=opt.exist_ok)
    opt.save_dir = str(save_path)

    try:
        train(opt.hyp, opt, device, callbacks)
    finally:
        if LOCAL_RANK != -1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
