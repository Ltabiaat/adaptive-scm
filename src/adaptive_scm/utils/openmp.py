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
    """Permit duplicate OpenMP runtimes on macOS and cap threads (no-op elsewhere).

    Sets ``KMP_DUPLICATE_LIB_OK=TRUE`` so a process that loads both XGBoost and
    PyTorch does not abort on the duplicate-runtime check, and sets
    ``OMP_NUM_THREADS=1`` so the two runtimes do not oversubscribe the cores and
    fight (which crawls rather than crashes). Neither is overwritten if already
    set. Must be called before importing either backend. The workloads here (a
    tiny MLP policy, single-series forecasts) gain nothing from OpenMP
    parallelism, so the thread cap costs no real speed. Does nothing off macOS.

    Returns:
        ``True`` if this call set ``KMP_DUPLICATE_LIB_OK``, ``False`` otherwise.
    """
    if sys.platform != "darwin":
        return False
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        return False
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return True
