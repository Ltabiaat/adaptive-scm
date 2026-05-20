"""Reproducibility helpers — global seeding across numpy, torch, and random.

Provides `set_global_seed`, the single entry point for setting RNG seeds
in every CLI script and test fixture. Torch is imported lazily so this
module is usable in Phase 1 before the deep-learning stack is installed
(per the optional dependency layout in pyproject.toml).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python's random, numpy, and (if installed) torch.

    Idempotent and side-effect-free beyond setting RNG state. Called once
    per replication by the experiment runner so that a (forecaster, policy,
    condition, seed) tuple is fully reproducible (PRD §7.3).

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: If True, also set torch.use_deterministic_algorithms
            and the cuBLAS workspace env var. Slower but bit-exact on GPU.
            Default False because TFT/PPO training need throughput more than
            bit-exactness across runs.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
