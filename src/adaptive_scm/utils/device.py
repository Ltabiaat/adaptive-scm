"""Compute-device resolution for Torch-backed models.

Provides a single ``resolve_device`` used by the PPO agent (and available to the
TFT forecaster) so device selection is explicit and consistent. The policy is
deliberately **cuda-or-cpu, never mps**: Apple's Metal (MPS) backend gives no
speed-up for the small MLP policies used here and has unsupported-operation gaps
that crash ``pytorch-forecasting``'s TFT, so it is avoided. This keeps an
Apple-Silicon machine on CPU (reliable) and a CUDA host on GPU (fast) with the
same code path.
"""

from __future__ import annotations


def resolve_device(prefer: str = "auto") -> str:
    """Return the Torch device string to use, never selecting MPS.

    With ``prefer="auto"`` returns ``"cuda"`` when a CUDA GPU is available and
    ``"cpu"`` otherwise. An explicit ``"cuda"`` or ``"cpu"`` is honoured when
    usable, falling back to ``"cpu"`` if CUDA is requested but unavailable. An
    explicit ``"mps"`` is downgraded to ``"cpu"`` with no error, since MPS is not
    supported by this project. Called by the PPO agent before constructing the
    SB3 model.

    Args:
        prefer: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        Either ``"cuda"`` or ``"cpu"``.
    """
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    choice = (prefer or "auto").lower()
    if choice == "cuda":
        return "cuda" if cuda_ok else "cpu"
    if choice in ("cpu", "mps"):
        return "cpu"
    # auto
    return "cuda" if cuda_ok else "cpu"
