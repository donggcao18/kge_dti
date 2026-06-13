import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(output_root: Path) -> None:
    for relative in [
        "auc",
        "model",
        "curve/roc",
        "curve/pr",
        "curve/roc_nfm",
        "curve/pr_nfm",
        "predictions",
        "ablation/auc",
        "ablation/curve/roc",
        "ablation/curve/pr",
        "ablation/predictions",
    ]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
    return device_arg


def parse_hidden_units(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())
