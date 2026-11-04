# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO26 utils package (YOLOv5-style layout)."""

import contextlib
import logging
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO26 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@contextlib.contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    if local_rank not in {-1, 0}:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class TryExcept(contextlib.ContextDecorator):
    """TryExcept class: wraps a function so that exceptions are printed instead of raised."""

    def __init__(self, msg=""):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


def set_logging(name="YOLO26", verbose=True):
    """Set up logging for the given name, with optional verbosity control."""
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(format="%(message)s", level=level)
    return logging.getLogger(name)


LOGGER = set_logging(verbose=True)
