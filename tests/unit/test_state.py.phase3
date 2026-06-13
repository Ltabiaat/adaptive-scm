"""Tests for the State dataclass passed to policies."""

from __future__ import annotations

import numpy as np

from adaptive_scm.policies.base import State


def make_state(on_hand: float = 10.0, pipeline: np.ndarray | None = None) -> State:
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
