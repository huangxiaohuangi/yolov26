# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental / model loading helpers (YOLOv5 风格)."""

from pathlib import Path

import torch


def attempt_load(weights, device=None, inplace=True, fuse=False):
    """Load a YOLO26 model from a *.pt checkpoint.

    Args:
        weights (str | list): checkpoint path(s). Only the first is used.
        device (torch.device | None): target device.
        inplace (bool): keep for YOLOv5 API compatibility.
        fuse (bool): fuse Conv+BN layers (inference speed-up).

    Returns:
        (nn.Module): model in eval mode with `names`, `nc`, `hyp`, `stride` attached.
    """
    from models.yolo import Model  # avoid circular import

    device = device or torch.device("cpu")
    w = weights[0] if isinstance(weights, (list, tuple)) else weights
    if not Path(w).is_file():
        raise FileNotFoundError(f"Checkpoint {w} not found")

    ckpt = torch.load(w, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise ValueError(f"Checkpoint {w} has neither 'ema' nor 'model' key")
    model = model.to(device).float()

    # Attach metadata (names / nc / hyp / stride)
    names = ckpt.get("names") or getattr(model, "names", None)
    model.names = names if names is not None else [str(i) for i in range(int(getattr(model, "nc", 80)))]
    model.nc = int(getattr(model, "nc", len(model.names)))
    opt = ckpt.get("opt") or {}
    model.hyp = opt.get("hyp") or getattr(model, "hyp", {}) or {}
    if not hasattr(model, "stride") or model.stride is None:
        model.stride = model.model[-1].stride

    model = model.eval()
    if fuse:
        model.fuse()
    return model


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models (YOLOv5 API compatibility)."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Forward pass through all models, averaging the outputs."""
        y = [module(x)[0] for module in self]
        return torch.cat(y, 1), None  # augmented inference, train
