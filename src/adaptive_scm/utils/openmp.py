"""macOS OpenMP coexistence guard.

XGBoost and PyTorch each bundle their own OpenMP runtime. On macOS, loading both
into one process aborts/segfaults unless the runtime is told to tolerate the
duplicate (D-4.7). Most of the codebase avoids this by never importing both in
one process (lazy forecaster imports; the suite isolates each cell in a
subprocess). The one unavoidable case is training a PPO agent on a *frozen
XGBoost* forecaster, where the XGBoost artifact and SB3's torch share a process.
``allow_duplicate_openmp`` enables coexistence there, on macOS only, and must run
before either backend is imported.
"""

from __future__ import annotations

import os
import sys


def allow_duplicate_openmp() -> bool:
    """Permit duplicate OpenMP runtimes on macOS (no-op elsewhere).

    Sets ``KMP_DUPLICATE_LIB_OK=TRUE`` when running on macOS and the variable is
    not already set, so a process that loads both XGBoost and PyTorch does not
    abort on the duplicate-runtime check. Must be called before importing either
    backend. On non-macOS platforms it does nothing (Linux shares one OpenMP
    runtime; CUDA hosts are unaffected).

    Returns:
        ``True`` if the flag was set by this call, ``False`` otherwise.
    """
    if sys.platform != "darwin":
        return False
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        return False
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return True
