"""Reproducibility seeding.

A single ``set_global_seed`` call seeds Python's ``random`` module, NumPy, and
(if installed) PyTorch. All experiment scripts call this at startup so the same
config + seed yields byte-identical results.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch from a single value.

    Sets ``PYTHONHASHSEED`` env var, seeds ``random.seed`` and ``numpy.random``,
    and if torch is importable also seeds CPU and CUDA RNGs and enables
    deterministic cuDNN. Called once per process at the entry point of every
    training/eval script.

    Args:
        seed: Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is optional at the data-pipeline layer; absence is fine.
        pass
