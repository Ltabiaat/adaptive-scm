"""Unit tests for the ``State`` dataclass passed to policies.

Validates the unified state shape adopted in Phase 3 (matching the PRD's PPO
observation): separate ``forecast_mean`` / ``forecast_std`` vectors, one-hot
``day_of_week``, ``time_index``, and the inventory-position / forecast-horizon
derived properties.
"""

from __future__ import annotations

import numpy as np

from adaptive_scm.policies.base import State


def make_state(on_hand: float = 10.0, pipeline: np.ndarray | None = None) -> State:
    """Build a ``State`` with sensible defaults for these tests.

    Args:
        on_hand: On-hand inventory in units.
        pipeline: Optional pipeline vector; defaults to ``[2, 3, 0]``.

    Returns:
        A populated :class:`State`.
    """
    if pipeline is None:
        pipeline = np.array([2.0, 3.0, 0.0], dtype=float)
    return State(
        on_hand=on_hand,
        pipeline=pipeline,
        forecast_mean=np.zeros(7),
        forecast_std=np.zeros(7),
        day_of_week=np.eye(7)[0],
        upcoming_events=np.zeros(7),
        time_index=0,
    )


def test_inventory_position_sums_on_hand_and_pipeline() -> None:
    s = make_state(on_hand=10.0, pipeline=np.array([2.0, 3.0, 0.0]))
    assert s.inventory_position == 15.0


def test_inventory_position_with_empty_pipeline() -> None:
    s = make_state(on_hand=5.0, pipeline=np.zeros(5))
    assert s.inventory_position == 5.0


def test_forecast_horizon_matches_vector_length() -> None:
    s = make_state()
    assert s.forecast_horizon == 7


def test_state_is_frozen() -> None:
    # The dataclass is frozen so a State cannot be mutated mid-step.
    import dataclasses

    import pytest

    s = make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.on_hand = 99.0  # type: ignore[misc]
