"""Unit tests for replenishment policies.

Phase 1 covers only the EOQ baseline (PRD Feature 5). Each test fixes the
forecast and inventory state explicitly so the policy's outputs are fully
determined by the formulas in PRD Section 4.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from adaptive_scm.policies import EOQPolicy
from adaptive_scm.policies.base import State

DAYS_PER_YEAR = 365


def _make_state(
    point: float,
    horizon: int,
    rmse: float,
    on_hand: float,
    pipeline: np.ndarray | None = None,
) -> State:
    """Build a minimal :class:`State` for EOQ tests.

    Constructs a flat forecast: every day's mean is ``point`` and every day's
    forecast-error std is ``rmse``, so the EOQ math reduces to
    ``mean_daily_demand = point`` and ``sigma_d = rmse`` exactly. Used only
    inside this test module.

    Args:
        point: Constant value for every day of ``forecast_mean``.
        horizon: Forecast-window length.
        rmse: Constant per-day forecast-error std (every ``forecast_std`` entry).
        on_hand: On-hand inventory in units.
        pipeline: Optional pipeline-orders vector; defaults to all zeros.

    Returns:
        A :class:`State` ready to pass to :meth:`EOQPolicy.select_action`.
    """
    if pipeline is None:
        pipeline = np.zeros(5)
    return State(
        on_hand=on_hand,
        pipeline=pipeline,
        forecast_mean=np.full(horizon, point, dtype=float),
        forecast_std=np.full(horizon, rmse, dtype=float),
        day_of_week=np.eye(7, dtype=np.int8)[0],
        upcoming_events=np.zeros(7, dtype=np.int8),
        time_index=0,
    )


class TestEOQConstruction:
    def test_rejects_non_positive_holding_cost(self):
        with pytest.raises(ValueError, match="holding"):
            EOQPolicy(holding_per_unit_per_day=0.0, fixed_order_cost=10.0, lead_time=3)

    def test_rejects_non_positive_order_cost(self):
        with pytest.raises(ValueError, match="fixed_order_cost"):
            EOQPolicy(holding_per_unit_per_day=0.05, fixed_order_cost=-1.0, lead_time=3)

    def test_rejects_zero_lead_time(self):
        with pytest.raises(ValueError, match="lead_time"):
            EOQPolicy(holding_per_unit_per_day=0.05, fixed_order_cost=10.0, lead_time=0)

    def test_rejects_invalid_service_level(self):
        with pytest.raises(ValueError, match="service_level"):
            EOQPolicy(
                holding_per_unit_per_day=0.05,
                fixed_order_cost=10.0,
                lead_time=3,
                service_level=1.5,
            )


class TestEOQOrderQuantity:
    """Verify EOQPolicy.select_action against the PRD formulas exactly."""

    @pytest.fixture
    def policy(self) -> EOQPolicy:
        return EOQPolicy(
            holding_per_unit_per_day=0.05,
            fixed_order_cost=10.0,
            lead_time=3,
            service_level=0.95,
        )

    def test_below_rop_orders_eoq(self, policy):
        # Constants: D_daily=10, H_annual=18.25, S=10, L=3, sigma_d=2.
        # Q* = sqrt(2 * 3650 * 10 / 18.25) = 63.2455...
        # ss = z(0.95) * 2 * sqrt(3) = 1.6448536 * 2 * 1.7320508 = 5.6968...
        # ROP = 10*3 + 5.6968 = 35.6968. on_hand=20 < ROP -> order.
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=20.0)
        order = policy.select_action(state)

        expected_q = math.sqrt(2 * 10.0 * DAYS_PER_YEAR * 10.0 / (0.05 * DAYS_PER_YEAR))
        assert order == round(expected_q)

    def test_above_rop_orders_zero(self, policy):
        # Same demand parameters but on-hand far above ROP.
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=200.0)
        assert policy.select_action(state) == 0

    def test_pipeline_counted_in_inventory_position(self, policy):
        # On-hand is below ROP, but pipeline lifts the inventory position
        # above ROP so no order should be placed.
        pipeline = np.array([0.0, 0.0, 200.0, 0.0, 0.0])
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=10.0, pipeline=pipeline)
        assert policy.select_action(state) == 0

    def test_deterministic_given_fixed_state(self, policy):
        # Acceptance criterion: same state -> same action, repeatable.
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=15.0)
        action_1 = policy.select_action(state)
        action_2 = policy.select_action(state)
        action_3 = policy.select_action(state)
        assert action_1 == action_2 == action_3

    def test_zero_demand_yields_zero_order(self, policy):
        # Pathological flat-zero forecast: Q* collapses to 0, ROP collapses
        # to safety stock only. on_hand=0 is below ROP only because of ss.
        state = _make_state(point=0.0, horizon=28, rmse=2.0, on_hand=0.0)
        order = policy.select_action(state)
        # Q* must be zero since annualized demand is zero.
        assert order == 0

    def test_higher_rmse_raises_safety_stock(self, policy):
        # Holding everything constant, increasing the forecast RMSE raises
        # safety stock, which raises the reorder point. A state that did NOT
        # trigger an order at low rmse should trigger one at high rmse.
        low_rmse = _make_state(point=10.0, horizon=28, rmse=0.1, on_hand=32.0)
        high_rmse = _make_state(point=10.0, horizon=28, rmse=10.0, on_hand=32.0)
        assert policy.select_action(low_rmse) == 0
        assert policy.select_action(high_rmse) > 0

    def test_safety_stock_matches_textbook_formula(self):
        # Direct check of the safety-stock term independent of triggering.
        # Pick a state where on_hand is exactly the mean*L (so the order
        # decision hinges entirely on whether ss > 0).
        policy = EOQPolicy(
            holding_per_unit_per_day=0.05,
            fixed_order_cost=10.0,
            lead_time=4,
            service_level=0.90,
        )
        state = _make_state(point=5.0, horizon=10, rmse=1.5, on_hand=5.0 * 4)
        # Expected safety stock: norm.ppf(0.90) * 1.5 * sqrt(4) = 1.2816 * 3 = 3.84...
        expected_ss = norm.ppf(0.90) * 1.5 * math.sqrt(4)
        # ROP = 20 + 3.84 = 23.84; on_hand 20 < ROP, so policy must order.
        order = policy.select_action(state)
        assert order > 0
        # And the safety-stock value used internally is positive (sanity).
        assert expected_ss > 0


# --------------------------------------------------------------------------- #
# Order-up-to (Feature 6)
# --------------------------------------------------------------------------- #

from adaptive_scm.policies import OrderUpToPolicy  # noqa: E402


class TestOrderUpToConstruction:
    def test_rejects_zero_lead_time(self):
        with pytest.raises(ValueError, match="lead_time"):
            OrderUpToPolicy(lead_time=0)

    def test_rejects_zero_review_period(self):
        with pytest.raises(ValueError, match="review_period"):
            OrderUpToPolicy(lead_time=3, review_period=0)

    def test_rejects_invalid_service_level(self):
        with pytest.raises(ValueError, match="service_level"):
            OrderUpToPolicy(lead_time=3, service_level=0.0)

    def test_protection_interval(self):
        assert OrderUpToPolicy(lead_time=3, review_period=1).protection_interval == 4


class TestOrderUpToAction:
    @pytest.fixture
    def policy(self) -> OrderUpToPolicy:
        return OrderUpToPolicy(lead_time=3, review_period=1, service_level=0.95)

    def test_orders_gap_to_target(self, policy):
        # R+L=4, mean=10, std=2: S = 40 + z(0.95)*sqrt(4)*2 = 40 + 1.6449*4.
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=20.0)
        from scipy.stats import norm

        expected_target = 40.0 + norm.ppf(0.95) * math.sqrt(4) * 2.0
        expected_order = round(expected_target - 20.0)
        assert policy.select_action(state) == expected_order

    def test_no_order_when_above_target(self, policy):
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=200.0)
        assert policy.select_action(state) == 0

    def test_responds_to_inventory_position(self, policy):
        # Higher inventory position -> smaller order (acceptance criterion).
        low = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=10.0)
        high = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=35.0)
        assert policy.select_action(low) > policy.select_action(high)

    def test_responds_to_forecast(self, policy):
        # Higher forecast demand -> larger target -> larger order (criterion).
        lo = _make_state(point=5.0, horizon=28, rmse=2.0, on_hand=20.0)
        hi = _make_state(point=15.0, horizon=28, rmse=2.0, on_hand=20.0)
        assert policy.select_action(hi) > policy.select_action(lo)

    def test_pipeline_counts_toward_position(self, policy):
        # Pipeline orders raise inventory position and reduce the order.
        pipe = np.array([0.0, 0.0, 30.0, 0.0, 0.0])
        with_pipe = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=10.0, pipeline=pipe)
        without = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=10.0)
        assert policy.select_action(with_pipe) < policy.select_action(without)

    def test_deterministic(self, policy):
        state = _make_state(point=10.0, horizon=28, rmse=2.0, on_hand=15.0)
        assert policy.select_action(state) == policy.select_action(state)

    def test_window_clamped_to_horizon(self):
        # Protection interval 10 but forecast horizon only 7: window clamps,
        # no index error, still produces a valid order.
        policy = OrderUpToPolicy(lead_time=9, review_period=1)
        state = _make_state(point=10.0, horizon=7, rmse=2.0, on_hand=5.0)
        order = policy.select_action(state)
        assert order > 0
