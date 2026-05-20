"""Tests for the global seeding utility."""

from __future__ import annotations

import random

import numpy as np
import pytest

from adaptive_scm.utils.seeding import set_global_seed


def test_set_global_seed_reproduces_numpy() -> None:
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_reproduces_random() -> None:
    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)
