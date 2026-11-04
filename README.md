# YOLO26（YOLOv5 风格代码库）

把 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 里的 **YOLO26** 重写成
[ultralytics/yolov5](https://github.com/ultralytics/yolov5) 那种"脚本 + yaml + utils"的组织方式：
`train.py` / `val.py` / `predict.py` 三个入口，`models/*.yaml` 定义网络，`utils/` 放数据、损失、指标。
**不依赖 `ultralytics` 包**（纯 `torch + opencv + pyyaml`），模型结构与官方实现逐比特对齐。
支持 **PyTorch 1.7 ~ 2.x**（高版本 API 全部做了版本探测与降级），详见[第 5 节](#5-环境与-pytorch-版本兼容17--2x)。

```
python train.py   --data data/coco128.yaml --cfg models/yolo26n.yaml --img 640 --batch 16 --epochs 100
python val.py     --weights runs/train/exp/weights/best.pt --data data/coco128.yaml
python predict.py --weights runs/train/exp/weights/best.pt --source data/images/bus.jpg
```

---

## 1. 两个上游仓库的差异分析

| 维度 | yolov5 | ultralytics（YOLO26 所在仓库） |
| --- | --- | --- |
| 组织方式 | 平铺脚本：`train.py` / `detect.py` / `val.py` + `models/` + `utils/` | Python 包：`ultralytics/`，通过 `YOLO("yolo26n.yaml").train()` 调用 |
| 模型定义 | `models/*.yaml` + `models/yolo.py:parse_model()` 生成 `nn.Sequential` | 同样是 yaml + `parse_model`，但藏在 `nn/tasks.py`，并按任务派生 `DetectionModel/SegmentationModel/...` |
| 检测头 | anchor-based：3 组 anchor、`no = nc + 5`、含 obj 分支 | anchor-free：`no = nc + 4*reg_max`，无 obj 分支；`reg_max=1` 时去掉 DFL |
| 标签分配 | `build_targets`：按 anchor 形状 + IoU 阈值匹配 | TAL（Task-Aligned Assigner）+ STAL（小目标感知、topk2 二次筛选） |
| 损失 | `ComputeLoss`：BCE(obj/cls) + CIoU + obj 平衡 | `v8DetectionLoss`（BCE + CIoU + L1）+ `E2ELoss`（双头 + ProgLoss 权重衰减） |
| NMS | 必须做 NMS | 可选 **NMS-free**：`end2end=True` 时用 one2one 分支直接输出 top-k |
| 训练入口 | `train.py`（warmup / EMA / 多尺度 / 早停 / evolve） | `engine/trainer.py` 基类 + 各任务 Trainer，配置驱动 |
| 权重格式 | `ckpt = {"model":..., "ema":..., "optimizer":...}` | 同样 dict，但模型对象是包内的类 |
| 依赖 | 现在也依赖 `ultralytics` 包提供部分 utils | 单体包，无外部框架依赖 |

**结论**：YOLOv5 的价值在于"文件即流程"，读代码成本低、改结构直观；ultralytics 的价值在于多任务/多后端。
本仓库保留前者的可读性，同时把后者的 YOLO26 结构、损失、训练策略 1:1 搬过来。

---

## 2. 本仓库结构

```
yolov26/
├── train.py                 # 训练入口（warmup / EMA / 多尺度 / 早停 / resume / results.csv）
├── val.py                   # 验证：mAP@0.5:0.95、PR/F1 曲线、混淆矩阵
├── predict.py               # 推理：图片/目录/视频/摄像头，--save-txt --save-crop --augment
├── requirements.txt
├── models/
│   ├── common.py            # Conv / DWConv / Bottleneck / C3 / C3k / C3k2 / C2PSA / SPPF / Detect
│   ├── yolo.py              # parse_model + DetectionModel（yaml → nn.Sequential）
│   ├── experimental.py      # attempt_load（加载 *.pt）
│   └── yolo26n.yaml ... yolo26x.yaml   # 5 个尺度
├── tools/
│   ├── verify_equivalence.py  # 与官方 ultralytics 逐比特对比（参数量/权重/前向/损失）
│   ├── check_torch_compat.py  # PyTorch 版本自检（1.7 ~ 2.x），--run 走一遍前向/反向
│   ├── check_ckpt_compat.py   # 验证官方 ckpt「类缺失」时的降级加载
│   ├── ddp_smoke.py           # CPU 双进程 DDP 冒烟测试（不依赖 torchrun）
│   ├── make_synth_dataset.py  # 生成合成检测数据集（快速验证收敛）
│   ├── overfit_test.py        # 固定单 batch 过拟合测试（验证梯度链路）
│   └── diag_loss.py           # 打印 loss 内部量（前景数、target_scores_sum 等）
├── utils/
│   ├── general.py           # 日志/yaml/box 转换/NMS/scale_boxes/路径
│   ├── torch_utils.py       # select_device / ModelEMA / EarlyStopping / smart_optimizer
│   ├── muon.py              # MuSGD 优化器（Muon + SGD）
│   ├── dataloaders.py       # LoadImagesAndLabels（mosaic/mixup/copy_paste）+ LoadImages
│   ├── augmentations.py     # letterbox / random_perspective / hsv / cutout
│   ├── tal.py               # make_anchors / dist2bbox / TaskAlignedAssigner（含 STAL）
│   ├── loss.py              # BboxLoss / v8DetectionLoss / E2ELoss / ComputeLoss
│   ├── metrics.py           # bbox_iou / ap_per_class / ConfusionMatrix / fitness
│   ├── plots.py             # Annotator / plot_labels / plot_results
│   └── callbacks.py
└── data/
    ├── coco128.yaml         # 自动下载的小数据集
    ├── custom.yaml          # 自定义数据集模板
    ├── hyps/hyp.scratch-{low,med,high}.yaml
    └── images/bus.jpg
```

### 与上游文件的对应关系

| 本仓库 | ultralytics 中的来源 | yolov5 中的对应 |
| --- | --- | --- |
| `models/common.py` | `nn/modules/conv.py` + `nn/modules/block.py` + `nn/modules/head.py:Detect` | `models/common.py` + `models/yolo.py:Detect` |
| `models/yolo.py` | `nn/tasks.py:parse_model / DetectionModel` | `models/yolo.py` |
| `utils/tal.py` | `utils/tal.py` | 无（yolov5 用 `build_targets`） |
| `utils/loss.py` | `utils/loss.py:v8DetectionLoss / E2ELoss / BboxLoss` | `utils/loss.py:ComputeLoss` |
| `utils/muon.py` | `utils/torch_utils.py` 里的 MuSGD 相关逻辑 | 无（只有 SGD/Adam） |
| `train.py` | `engine/trainer.py` + `models/yolo/detect/train.py` | `train.py` |
| `val.py` | `engine/validator.py` + `models/yolo/detect/val.py` | `val.py` |
| `predict.py` | `engine/predictor.py` + `models/yolo/detect/predict.py` | `detect.py` |
| `utils/dataloaders.py` | `data/dataset.py` + `data/build.py` + `data/augment.py` | `utils/dataloaders.py` |
| `models/yolo26*.yaml` | `cfg/models/26/yolo26.yaml` | `models/yolov5*.yaml` |

---

## 3. YOLO26 的模型结构（`models/yolo26n.yaml`）

```
                 from  n    params  module
  0                -1  1       464  Conv[3,16,3,2]                 # P1/2
  1                -1  1      4672  Conv[16,32,3,2]                # P2/4
  2                -1  1      6640  C3k2[32,64,1,False,0.25]
  3                -1  1     36992  Conv[64,64,3,2]                # P3/8
  4                -1  1     26080  C3k2[64,128,1,False,0.25]
  5                -1  1    147712  Conv[128,128,3,2]              # P4/16
  6                -1  1     87040  C3k2[128,128,1,True]
  7                -1  1    295424  Conv[128,256,3,2]              # P5/32
  8                -1  1    346112  C3k2[256,256,1,True]
  9                -1  1    164608  SPPF[256,256,5,3,True]
 10                -1  1    249728  C2PSA[256,256,1]               # 主干末端全局注意力
 11..16  PAN 上采样 + Concat + C3k2                               # P3 输出
 17..19  Conv 下采样 + Concat + C3k2                              # P4 输出
 20..22  Conv 下采样 + Concat + C3k2[...,True,0.5,True]           # P5 输出（带 PSABlock）
 23      [16,19,22]  1    309656  Detect[80, 1, True, [64,128,256]]
```

要点：

1. **anchor-free + 去 DFL**：`reg_max: 1`，回归分支只输出 4 个通道（ltrb），
   `Detect.decode_bboxes` 用 `dist2bbox` 解码；不再有 anchor 匹配与 obj 分支。
2. **双头（NMS-free）**：`end2end: True` 时额外深拷贝一份 `one2one_cv2/cv3`。
   训练时两个头都算 loss；推理时只走 one2one，直接 top-k 输出 `(bs, 300, 6)`，**不需要 NMS**。
3. **C3k2**：CSP + 内部块可选 `Bottleneck` / `C3k` / `Bottleneck+PSABlock`。
   注意官方对 **m/l/x** 尺度会把 `c3k` 强制设为 `True`（`parse_model` 里 `args[3:4] = [True]`），
   本仓库同样实现，因此参数量与官方完全一致。
4. **C2PSA**：主干最后一层，`PSABlock`（带位置编码的 MHA + FFN，残差）。
5. **SPPF**：`shortcut=True`，输出与输入相加（YOLO26 版）。

5 个尺度与官方参数量对比（`python models/yolo.py --cfg models/yolo26n.yaml`）：

| scale | depth | width | max_ch | 本仓库参数量 | 官方参数量 |
| --- | --- | --- | --- | --- | --- |
| n | 0.50 | 0.25 | 1024 | 2,572,280 | 2,572,280 |
| s | 0.50 | 0.50 | 1024 | 10,009,784 | 10,009,784 |
| m | 0.50 | 1.00 | 512 | 21,896,248 | 21,896,248 |
| l | 1.00 | 1.00 | 512 | 26,299,704 | 26,299,704 |
| x | 1.00 | 1.50 | 512 | 58,993,368 | 58,993,368 |

---

## 4. 损失与标签分配

### 4.1 TAL + STAL（`utils/tal.py`）

* 对齐度 `t = score^0.5 * IoU^6`，每个 GT 取 topk 个 anchor（one2many: `topk=10`）。
* **STAL**（小目标感知）：
  * `select_candidates_in_gts` 中，把边长 < `stride` 的 GT 框放大到 `stride`（默认取中间层 stride=16），
    避免小目标在 P3 上也选不到正样本；
  * `select_highest_overlaps` 中，`topk2 != topk` 时按对齐度再做一次 topk 筛选
    （one2one 分支 `topk=7, topk2=1`），保证每个 GT 只留 1 个正样本，这是 NMS-free 的前提。

### 4.2 box 回归（`utils/loss.py:BboxLoss`）

* 定位：`CIoU`；
* 回归：`reg_max=1` 时 **不做 DFL**，改为把 ltrb 按 stride 缩放、再按图像宽高归一化后算 `L1`；
* 损失项：`box_loss / cls_loss / l1_loss`（与官方 `loss_names` 一致）。

### 4.3 双头 + ProgLoss（`E2ELoss`）

```
loss = 0.8→0.1 的权重 * one2many_loss + (1 - 该权重) * one2one_loss
```

权重随 epoch 线性衰减（`o2m: 0.8 → 0.1`），前期靠 one2many 提供密集监督，
后期把优化重心交给部署用的 one2one 分支——这就是 YOLO26 的 progressive loss balancing。

---

## 5. 环境与 PyTorch 版本兼容（1.7 ~ 2.x）

代码里所有高版本 API 都做了**版本探测 + 降级**，同一份源码在 torch 1.7 / 1.8 / 2.x 上都能跑，
不需要改任何一行。

### 5.1 版本矩阵

| 组件 | 推荐版本 | 说明 |
| --- | --- | --- |
| Python | **3.7 / 3.8 / 3.9** | torch 1.7 的官方 wheel 覆盖 CPython 3.6~3.9；源码本身也兼容 3.7 语法 |
| PyTorch | **1.7.1+** | 1.7.0 也可，1.7.1 修了几个 1.7.0 的 bug |
| torchvision | **0.8.1** | 必须与 torch 1.7 配套 |
| NumPy | **1.18.5 ~ 1.24** | 必须 `<2.0`，torch 1.7 与 numpy 2.x 的 C-ABI 不兼容 |
| OpenCV | 4.6+ | 仅用 `cv2.imread/resize/warpAffine` 等基础接口 |

> 代码里刻意避开了 Python 3.8+ 的语法/API：没有 walrus（`:=`）、没有 `list[int]` 泛型下标、
> 没有 `Path.unlink(missing_ok=)`（改用 `contextlib.suppress(FileNotFoundError)`）、
> 没有 `math.prod` / `functools.cached_property`。所以在 3.7 上可以直接跑。

安装（以 CUDA 11.0 为例，CPU 版把 `+cu110` 换成 `+cpu`）：

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.1+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

> 如果你现在装的是 torch 2.x 想降到 1.7，**必须换一个 Python ≤3.9 的环境**（conda 新建最省事）：
> `conda create -n yolo26 python=3.9 && conda activate yolo26`。

### 5.2 为兼容 1.7 改动了什么

| 位置 | 高版本写法 | 1.7 的问题 | 本仓库的处理 |
| --- | --- | --- | --- |
| `utils/general.py:torch_load` | `torch.load(..., weights_only=False)` | 没有该参数，会被当成 pickle 参数报 `TypeError` | 按 `torch>=1.13` 判断，旧版直接 `torch.load(f, map_location=...)` |
| `utils/tal.py:make_anchors` | `torch.meshgrid(a, b, indexing="ij")` | 没有 `indexing` 参数 | `try/except TypeError` 回退到默认 `ij` 行为 |
| `utils/general.py:init_seeds` | `torch.use_deterministic_algorithms(True, warn_only=True)` | 1.8 才有 | `hasattr` 判断，旧版只设 `cudnn.deterministic` |
| `utils/torch_utils.py:select_device` | `torch.backends.mps.is_available()` | `torch.backends.mps` 1.12 才有，直接 `AttributeError` | 模块级 `MPS_AVAILABLE` 探测 |
| `utils/torch_utils.py:smart_optimizer` | `getattr(torch.optim, "NAdam")` | `NAdam` 1.10 才有 | `getattr(..., None)` 为空时回退 `Adam` 并打警告 |
| `utils/torch_utils.py:ModelEMA.update` | `v.dtype.is_floating_point` | 属性在旧版不稳定 | 改用 `torch.is_floating_point(v)` |
| `models/yolo.py:BaseModel._apply` | `super()._apply(fn, recurse)` | `recurse` 参数 2.0 才有 | 按版本单/双参数调用 |
| `train.py` AMP | `torch.amp.GradScaler("cuda", ...)` | `torch.amp` 2.4 才有 | 按版本分支，旧版走 `torch.cuda.amp.GradScaler` |

其余部分（`C3k2 / C2PSA / SPPF / Detect`、TAL、`baddbmm`、`topk`、`one_hot`、
`clip_grad_norm_`、`LambdaLR`、`torchvision.ops.nms`）在 1.7 里都已存在，无需改动。

### 5.2.1 加载官方 `yolo26n.pt` 时类缺失怎么办

这是独立于 torch 版本的一个坑：官方 `.pt` 是 `torch.save({"model": DetectionModel(...)})`，
pickle 里存的是**类全名** `ultralytics.nn.modules.block.C3k2`。如果本机 ultralytics 版本较旧，
反序列化阶段就会报：

```
AttributeError: Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>
```

此时权重张量本身完好，只是缺一个类定义。`utils/general.py:torch_load` 会自动兜底：

1. 捕获 `AttributeError` / `ModuleNotFoundError`；
2. 用自定义 `_CompatUnpickler` 把**无法导入的类**（仅限 `ultralytics.*` / `models.*` / `utils.*` 前缀）
   替换成占位类 `_CkptStub`，`torch.*` 等关键模块仍正常导入；
3. 占位实例的 `__dict__` 里 `_parameters` / `_buffers` / `_modules` 结构与真实 `nn.Module` 一致，
   递归遍历即可拿到完整 `state_dict`；
4. 包装成 `_CkptModel` 返回，`train.py` 的 `ckpt["model"].yaml` / `.float().state_dict()` 照常工作；
5. `attempt_load` 检测到 `_CkptModel` 时，用 ckpt 里保存的 `model.yaml` 重建本仓库模型再灌权重。

实测（`tools/check_ckpt_compat.py`）：`Transferred 708/708 items`，权重 100% 迁移。

> 注意：ckpt 里的 `model.yaml` 是**未展开**的原始 yaml（含 `scales` + `scale`，不含 `depth_multiple`）。
> 从 dict 构建模型时必须用 yaml 自带的 `scale`，不能从文件名推断 —— 否则 `model.yaml` 的
> 末字符是 `l`，n/s 尺度会被当成 l 尺度，参数量全错。

### 5.3 自检

装好环境后先跑这一条，它会逐项打印哪些 API 走新分支、哪些走降级：

```bash
python tools/check_torch_compat.py --run
```

`--run` 会额外构造 `yolo26n`、跑一次 前向 → 损失 → 反向 → 优化器 step，
确认整条链路在你的 torch 版本上真的能走通。输出示例：

```
✅ 全部必需项通过（torch 1.7.1+cu110, torchvision 0.8.1+cu110）。
   其中 4 项为可选 API，已自动走降级路径：
   torch.load(weights_only=)、torch.meshgrid(indexing=)、torch.use_deterministic_algorithms、torch.optim.NAdam
```

再跑这一条，验证「官方 ckpt 类缺失」时的降级加载（不需要真实的 `yolo26n.pt`，脚本会自己造假 ckpt）：

```bash
python tools/check_ckpt_compat.py
```

它依次打印：直接 `torch.load` 的报错 → 降级加载 → 提取到的每个张量 → 重建模型 → `attempt_load` 推理。
最后一行是 `✅ 降级加载链路正常` 就说明你直接 `--weights yolo26n.pt` 也能跑。

### 5.4 1.7 环境下的已知限制

* **AMP 不生效**：CPU 训练自动关闭；CUDA 上 1.7 的 `GradScaler` 可用，但 `--amp` 的收益不如 2.x 明显。
* **MuSGD 的 Muon 分量在 CPU 上只作用于 2D 权重和 1×1 卷积**（`muon_scope="auto"`），
  因为 3×3 卷积核的 Newton-Schulz 在 CPU 上比前向还慢；GPU 上自动切到 `"all"`。
  Newton-Schulz 在 GPU 用 bf16、CPU 用 fp32（1.7 的 CPU bf16 没有向量化实现）。
* **没有 `torch.inference_mode()`**：`predict.py` 用 `torch.no_grad()`，功能一致、速度略慢。
* **`torch.profiler`**（1.8+）未使用，`Profile` 类基于 `torch.cuda.Event` + `time.time()` 自行实现。
* 导出（ONNX/TensorRT）不在本仓库范围内，与 torch 版本无关。

---

## 6. 用法

### 6.1 训练

```bash
# 从头训练（COCO128，自动下载）
python train.py --data data/coco128.yaml --cfg models/yolo26n.yaml --img 640 --batch 16 --epochs 100

# 断点续训
python train.py --weights runs/train/exp/weights/last.pt --resume

# 多卡 DDP（2 卡；--batch 是「单卡」batch，全局 = 8*2 = 16）
torchrun --nproc_per_node=2 train.py --data data/coco128.yaml --cfg models/yolo26n.yaml \
    --batch 8 --epochs 100 --device 0,1

# 常用开关
--optimizer MuSGD|SGD|Adam|AdamW   # 默认 MuSGD（YOLO26 默认优化器）
--muon-scope auto|all|linear|none  # MuSGD 里 Muon 的作用范围，见下
--cos-lr --multi-scale --cache ram --rect --amp --freeze 10 --patience 50
```

**`--muon-scope` 怎么选**：Muon 对每个 2D/4D 权重做 5 次 Newton-Schulz 正交化，
代价是 `O(5·n·m·min(n,m))`。yolo26x 有 190 个这样的权重（含 768×19200 的 5×5 卷积核），
单卡上第一次 `optimizer.step()` 可能要几十秒甚至更久，看起来像"卡住"。

| 取值 | 行为 | 适用场景 |
| --- | --- | --- |
| `auto`（默认） | GPU→`all`，CPU→`linear` | 与官方一致；多卡 A100 级别 |
| `all` | 所有 2D/4D 权重都做 NS | 官方行为，单卡 yolo26x 会很慢 |
| `linear` | 只对 2D 权重和 1×1 卷积做 NS | **单卡/小显存推荐**，计算量约 1/10 |
| `none` | 不进 Muon 组，MuSGD 退化为纯 SGD | 只想快速跑通 |

还可以用环境变量给单个矩阵设上限（超过就只做 Frobenius 归一化，跳过 NS）：

```bash
YOLO_MUON_MAX_ELEMS=2000000 python train.py ...   # 200 万元素以上的矩阵跳过 NS
YOLO_MUON_BATCH_ELEMS=20000000 python train.py ...  # 限制 padding 后的批次大小，防显存爆
```

训练开始后第一行会打印 `首次 optimizer.step() 耗时 x.xx s`，超过 3 秒会附带加速建议。

产物：`runs/train/exp/{weights/{last,best}.pt, results.csv, results.png, hyp.yaml, opt.yaml}`

### 6.2 验证

```bash
python val.py --weights runs/train/exp/weights/best.pt --data data/coco128.yaml --img 640
```

### 6.3 推理

```bash
python predict.py --weights runs/train/exp/weights/best.pt --source data/images/bus.jpg
python predict.py --weights ... --source imgs/ --save-txt --save-conf
python predict.py --weights ... --source video.mp4 --view-img
python predict.py --weights ... --source 0          # 摄像头
python predict.py --weights ... --source imgs/ --augment   # 水平翻转 TTA
```

产物：`runs/detect/exp/*.jpg`、`labels/*.txt`、`crops/<cls>/*.jpg`

### 6.4 自定义数据集

```
datasets/mydata/
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt      # class x_center y_center width height（归一化）
```

复制 `data/custom.yaml` 改 `path/nc/names` 即可。

### 6.5 多卡 DDP

```bash
# 2 卡（--batch 是单卡 batch，全局 = 8 × 2 = 16）
torchrun --nproc_per_node=2 train.py --data data/coco128.yaml --cfg models/yolo26n.yaml \
    --batch 8 --epochs 100 --device 0,1

# 4 卡
torchrun --nproc_per_node=4 train.py ... --batch 4 --device 0,1,2,3
```

要点：

| 项 | 说明 |
| --- | --- |
| `--device` | 填**全部**卡（如 `0,1`），用于设置 `CUDA_VISIBLE_DEVICES`；每个进程再用 `LOCAL_RANK` 绑自己的卡 |
| `--batch` | **单卡** batch，必须能被卡数整除 |
| 学习率 | 不用改。全局 batch 变大后 `accumulate` 会自动降为 1，`weight_decay` 按全局 batch 缩放 |
| `--workers` | 是**每进程**的 worker 数，总 worker = `workers × 卡数` |
| 验证 | 只在 rank 0 做，用完整的验证集（`val_loader` 不分片） |
| 保存 / 日志 | 只在 rank 0 写 `results.csv`、`*.pt`、曲线图；其余 rank 日志级别降到 ERROR |
| 早停 | `stop` 通过 `all_reduce` 同步，避免各 rank 步数不一致导致集合通信死锁 |
| `--image-weights` | **不兼容**，会直接报错 |
| `--sync-bn` | 可用，`SyncBatchNorm` 会在 DDP 包装前转换 |

如果 `torchrun` 报 `use_libuv was requested but PyTorch was built without libuv support`（常见于某些 Windows/自编译 torch），
用环境变量绕过：

```bash
YOLO_DDP_INIT="tcp://127.0.0.1:29532?rank=$RANK&world_size=$WORLD_SIZE" torchrun ...
```

没有 GPU 的机器想验证 DDP 逻辑，可以跑 `python tools/ddp_smoke.py`（2 进程 + gloo + CPU）。

---

## 7. 已验证（与官方实现逐比特对比）

一条命令跑完全部对照（需要本地安装 `ultralytics`，仅用于对比，不是运行依赖）：

```bash
python tools/verify_equivalence.py            # 默认 5 个尺度 / nc=80 / 640
python tools/verify_equivalence.py --imgsz 320 --nc 2
```

| 验证项 | 方法 | 结果 |
| --- | --- | --- |
| 参数量 | 5 个尺度 vs `yolo26.yaml` 注释值 | 全部一致（见上表） |
| 权重兼容 | `mine.load_state_dict(ref.state_dict(), strict=True)` | 通过（键名/形状完全对应） |
| 前向（训练态） | 同一权重、同一输入，比较 `one2many/one2one` 的 `boxes/scores` | 最大误差 **0.0** |
| 前向（推理态） | `end2end` top-k 输出 `(bs, 300, 6)` | 最大误差 **0.0** |
| 前向（非 end2end） | one2many 输出 `(bs, na, 4+nc)` | 最大误差 **0.0**（仅内存布局不同） |
| 损失 | `ComputeLoss` vs `E2ELoss`（total + box/cls/l1） | 最大误差 **0.0** |
| 标签分配 | `TaskAlignedAssigner`（topk=10 与 topk=7/topk2=1） | 五项输出全部逐比特相同 |
| torch 版本兼容 | `python tools/check_torch_compat.py --run` | 必需 API 全绿，可选 API 自动降级 |
| 端到端 | 合成数据集训练 → val → predict | 见下 |

### 端到端冒烟训练结果

数据：`tools/make_synth_dataset.py` 生成的 192 训练 / 48 验证、256×256、2 类彩色块。
配置：`yolo26n`、`imgsz=192`、`batch=16`、45 epoch、CPU、默认 `MuSGD`。

| 指标 | 值 |
| --- | --- |
| mAP@0.5 | **0.861**（best，epoch 40）；`best.pt` 单独验证 0.855 |
| mAP@0.5:0.95 | **0.655** |
| train box_loss | 3.02 → 0.97 |
| train cls_loss | 3.83 → 1.03 |
| train l1_loss | 0.0639 → 0.0159 |

复现：

```bash
python tools/make_synth_dataset.py --root /tmp/synth --train 192 --val 48 --size 256
python train.py --data /tmp/synth/synth2.yaml --cfg models/yolo26n.yaml \
                --hyp data/hyps/hyp.scratch-low.yaml --epochs 45 --batch 16 --img 192 --device cpu
python val.py   --weights runs/train/exp/weights/best.pt --data /tmp/synth/synth2.yaml --img 192
python predict.py --weights runs/train/exp/weights/best.pt --source /tmp/synth/images/val
```

> 单 batch 过拟合测试（`tools/overfit_test.py`）可以单独确认梯度链路：
> 总损失 60.2 → 11.7（100 步），box_loss 3.10 → 0.49。

---

## 8. 与官方实现的差异 / 未覆盖部分

* **只做检测任务**：`-seg / -pose / -obb / -cls` 未实现（结构上可照 `Detect` 的写法扩展）。
* **DDP 已实现**（`torchrun` / `torch.distributed.launch` 均可），但**不支持 `--image-weights`**
  （加权采样与 `DistributedSampler` 冲突）。验证 / 保存 / 写日志只在 rank 0 做，
  早停标志通过 `all_reduce` 同步，避免各 rank 步数不一致导致集合通信死锁。
* **AMP 仅 CUDA**：CPU 上自动关闭。
* **MuSGD**：数学行为与官方 `ultralytics/optim/muon.py` 一致（动量 EMA + Newton-Schulz 五次迭代，
  `muon=0.2 / sgd=1.0` 双分量），参数分组也照 `build_optimizer` 拆成
  weight / bn / bias / muon 四组，并给 `Detect.cv3` 分类头 `lr×3`。
  **差异**：官方对 `ndim in {2,4}` 全部启用 Muon；本实现在 CPU 上默认只对 2D 权重与 1×1 卷积启用
  （3×3 卷积核的 Newton-Schulz 在 CPU 上比整个前向还慢），CUDA 上等价于官方行为。
* **`accumulate` 小数据集保护**：官方 `nbs=64`，`batch<64` 时会累积梯度。若一个 epoch 的 batch 数
  少于累积步数（例如 64 张图、batch=8，累积 8 步），整个训练只会更新十几次参数。
  本仓库把 `accumulate` 上限钳到 `nb`（每个 epoch 至少优化一次）并打印提示。
* **`end2end` 默认值**：官方模型构造完成后 `Detect.end2end` 属性为 `False`，由 validator/exporter
  在 `nms=False` 时显式打开；本仓库的 `Detect` 直接沿用 yaml 的 `end2end: True`，
  因此 `predict.py` / `val.py` 默认走 NMS-free 路径。
* **无 AutoAnchor**：anchor-free 不需要；`--noautoanchor` 参数已移除。
* **导出**（ONNX/TensorRT 等）未包含：`predict.py` 只支持 PyTorch 权重。
* `models/experimental.py` 只支持加载本仓库保存的 `*.pt`（官方 `yolo26n.pt` 的类不同，不能直接 `attempt_load`，
  但权重张量可以通过 `state_dict` 互相转换，见第 7 节）。

---

## 9. 常见问题

**Q: 训练启动后卡在 `Epoch GPU_mem box_loss ...` 后面，没有进度条？**
A: 按顺序排查这三点：

1. **MuSGD 的第一次 `optimizer.step()`**（最常见）。warmup 期间 `accumulate=1`，所以**第一个 batch 就会 step**；
   `muon_scope=all` 时 yolo26x 的 190 个卷积核都要做 5 次 Newton-Schulz，单卡可能要几十秒到几分钟，
   表现就是进度条停在 0%。日志里会打印 `首次 optimizer.step() 耗时 x.xx s` 帮你确认。
   解决：`--muon-scope linear`（推荐，计算量约 1/10）或 `--optimizer SGD`。
2. **数据加载**。`--workers 8` 遇到慢磁盘、或数据集里混了超大原图时，第一个 batch 要解码 `4×batch` 张图。
   用 `--workers 0` 跑一次对比即可判断。
3. **显存不足**。`640×8` 的 yolo26x 激活可能 10GB+，显存打满时 CUDA 会反复换页（不报错但极慢）。
   `nvidia-smi` 观察显存占用，降低 `--batch` 或 `--imgsz`。

另外注意：只写 `--device 0,1` 而不用 `torchrun` 启动时，**只会用 cuda:0**（没有 DDP 环境变量），启动时会打警告。
多卡请用 `torchrun --nproc_per_node=2 train.py ... --device 0,1`，见 [6.5 节](#65-多卡-ddp)。

**Q: 训练 loss 一直不降怎么办？**
A: 按这三步排查（本仓库的三个 `tools/` 脚本就是为此准备的）：
1. `python tools/overfit_test.py SGD 200` —— 固定单 batch 能否把 loss 打到接近 0？打不下去说明梯度链路有问题；
2. 看 `accumulate` 是否过大：`nb`（每 epoch 的 batch 数）< `accumulate` 时，一个 epoch 可能一次参数都不更新；
3. 看 `--lr0` 与 `--warmup-epochs`：warmup 太长而总步数很短时，学习率基本一直贴着 0。

**Q: val 的 mAP 一直是 0？**
A: 最常见的两个原因：
1. 标签和预测框不在同一坐标系。dataloader 输出的标签是**归一化** xywh，必须
   `targets[:, 2:] *= torch.tensor((w, h, w, h))` 转成像素后再算 IoU（`val.py` 已处理）；
2. `data.yaml` 的 `nc` 与 `names` 数量不一致，或 `--single-cls` 与多类数据混用。

**Q: `TypeError: unsupported operand type(s) for +=: 'int' and 'str'`（`cache_labels` 里）？**
A: `verify_image_label` 的计数器初值写错成 `nf = im_file`（字符串），遇到**损坏样本**走异常分支时就炸。
已修复（`cache_version` 提到 `1.1`，旧缓存会自动失效重建）。
顺带说明：数据集里有非图片文件、标签列数 ≠ 5、坐标越界、负值时，现在会打警告并**跳过该样本**，不会中断训练：
```
train: WARNING ⚠️ xxx.jpg: ignoring corrupt image/label: labels require 5 columns, 6 columns detected
```

**Q: Windows 上每次都重新扫描数据集，`New cache created` 之后下次还是 `Scanning`？**
A: `Path.rename()` 在 Windows 上不允许覆盖已存在的文件（`WinError 183`），缓存写不进去。
已改用 `os.replace()`，跨平台都能覆盖。如果你在旧版本遇到过，删掉 `labels/*.cache` 重跑一次即可。

**Q: 训练中途 loss 突然飙高又自己恢复？**
A: 小数据集 + `lr0=0.01` 时可能出现过冲（本例 epoch 11 附近 `val/loss` 冲到 1200，两轮后恢复）。
想更稳可以 `--lr0 0.005`、加大 `--batch`（让 `accumulate` 接近 1），或改用 `--optimizer SGD` 对照。

**Q: 推理输出没有经过 NMS，会不会重复框？**
A: `end2end=True` 的模型靠 one2one 分支 + top-k 抑制重复，正常情况下不需要 NMS。
如需传统流程，把 yaml 里 `end2end: False`，模型会输出 `(bs, na, 4+nc)` 的预测，`predict.py` 会自动走 NMS。

**Q: 想改成自己的类别数？**
A: 不用改 yaml，`train.py` 会用 `data.yaml` 的 `nc` 覆盖（`Model(cfg, nc=nc)`）。

**Q: 在 torch 1.7 上报 `TypeError: load() got an unexpected keyword argument 'weights_only'`？**
A: 说明用的是旧版代码。当前版本已把所有 `torch.load` 收敛到 `utils/general.py:torch_load`，
按版本自动决定要不要传 `weights_only`。先跑 `python tools/check_torch_compat.py` 确认没有 `[FAIL]` 项。

**Q: `--weights yolo26n.pt` 报 `AttributeError: Can't get attribute 'C3k2' on <module 'ultralytics.nn.modules.block'>`？**
A: 官方 `.pt` 里 pickle 的是 ultralytics 的类，本机 ultralytics 版本太旧没有 `C3k2`。
当前版本的 `torch_load` 会自动降级为「只取 state_dict」模式，权重照样 100% 迁移（实测 `Transferred 708/708`）。
若仍失败，跑 `python tools/check_ckpt_compat.py` 看是哪一步断了。
也可以临时升级 ultralytics 到含 YOLO26 的版本，或让同事在那边导出纯权重：
`torch.save(model.state_dict(), "yolo26n_sd.pt")`。

**Q: 装不上 torch 1.7，pip 说 "No matching distribution"？**
A: torch 1.7 的 wheel 只发布到 **CPython 3.9**，3.10+ 一律装不上。新建一个 Python 3.8/3.9 的 conda 环境即可：
`conda create -n yolo26 python=3.9 -y && conda activate yolo26`，再按第 5.1 节的命令安装。

**Q: torch 1.7 下 import 时报 numpy 相关的 ABI 错误？**
A: numpy 2.x 与 torch 1.7 的 C-ABI 不兼容，必须降级：`pip install "numpy<2"`。

**Q: 1.7 上跑得比 2.x 慢？**
A: 主要差在 Muon 的正交化（Newton-Schulz）。CPU 上默认已限制为 2D 权重 + 1×1 卷积；
GPU 上会用 bf16 加速。想完全避开这部分开销可用 `--optimizer SGD` 或 `--optimizer AdamW`。

---

## License

代码遵循上游 **AGPL-3.0**（与 ultralytics/ultralytics、ultralytics/yolov5 一致），
结构、算法与超参来源于这两个仓库，请遵守其许可协议。
