"""Unit tests for compute-device resolution.

Confirms the cuda-or-cpu policy that keeps Apple-Silicon machines on CPU: MPS is
never selected, and CUDA is requested only when actually available.
"""

from __future__ import annotations

from adaptive_scm.utils.device import resolve_device


class TestResolveDevice:
    def test_never_returns_mps(self):
        for prefer in ("auto", "cpu", "cuda", "mps"):
            assert resolve_device(prefer) in ("cpu", "cuda")

    def test_mps_downgrades_to_cpu(self):
        assert resolve_device("mps") == "cpu"

    def test_cpu_stays_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_is_cpu_without_cuda(self):
        # The test host has no CUDA GPU, so auto must resolve to cpu.
        import torch

        if not torch.cuda.is_available():
            assert resolve_device("auto") == "cpu"
            assert resolve_device("cuda") == "cpu"

    def test_unknown_string_falls_back_to_auto(self):
        assert resolve_device("something-odd") in ("cpu", "cuda")
