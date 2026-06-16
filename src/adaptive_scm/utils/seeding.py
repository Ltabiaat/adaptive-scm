"""Reproducibility seeding.

A single ``set_global_seed`` call seeds Python's ``random`` module, NumPy, and
(if installed) PyTorch. All experiment scripts call this at startup so the same
config + seed yields byte-identical results.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if already loaded) PyTorch from a single value.

    Sets ``PYTHONHASHSEED``, seeds ``random`` and ``numpy.random``, and seeds
    PyTorch's CPU/CUDA RNGs **only if torch is already imported** (checked via
    ``sys.modules``). It deliberately does not ``import torch`` itself: doing so
    would pull torch's OpenMP runtime into a process that may also load XGBoost,
    which segfaults on macOS (D-4.7 / D-4.8). Torch-based components (TFT, PPO)
    seed torch where they load it, so reproducibility is preserved without this
    helper forcing the import. Called once per process at each script's entry.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    seed_torch_if_loaded(seed)


def seed_torch_if_loaded(seed: int) -> bool:
    """Seed PyTorch's RNGs, but only if torch is already imported.

    Checks ``sys.modules`` rather than importing torch, so a torch-free process
    (e.g. an ARIMA or XGBoost run) is never forced to load torch's OpenMP runtime
    (D-4.8). Torch components call this after they import torch to lock in
    reproducibility.

    Args:
        seed: Non-negative integer seed.

    Returns:
        ``True`` if torch was loaded and seeded, ``False`` otherwise.
    """
    if "torch" not in sys.modules:
        return False
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True
