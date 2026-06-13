"""Unit tests for the simulation environment and disruption wrappers.

Covers PRD Features 7 and 8: Gymnasium compliance, the order/arrival/demand/
cost mechanics, lost-sales behavior, the structured-state projection, and the
two disruption wrappers. Uses deterministic synthetic episodes so trajectories
are fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_scm.policies import EOQPolicy, OrderUpToPolicy
from adaptive_scm.simulation import (
    DemandSpikeWrapper,
    EnvConfig,
    EpisodeData,
    InventoryEnv,
    LeadTimeDisruptionWrapper,
)
from adaptive_scm.simulation.environment import ACTION_MULTIPLIERS, FORECAST_WINDOW

_N = 28
_W = FORECAST_WINDOW


def _episode(demand_value: float = 10.0, n: int = _N) -> EpisodeData:
    """Build a flat deterministic episode for tests.

    Args:
        demand_value: Constant realized demand per day.
        n: Episode length.

    Returns:
        An :class:`EpisodeData` with constant demand and forecast.
    """
    return EpisodeData(
        demand=np.full(n, demand_value),
        forecast_mean=np.full(n + _W, demand_value),
        forecast_std=np.full(n + _W, 2.0),
        day_of_week=(np.arange(n) % 7),
        events=np.zeros(n + _W),
    )


def _config(**overrides) -> EnvConfig:
    """Build an EnvConfig with test-friendly defaults.

    Args:
        **overrides: Fields to override on the default config.

    Returns:
        An :class:`EnvConfig`.
    """
    base = dict(episode_length=_N, mean_daily_demand=10.0)
    base.update(overrides)
    return EnvConfig(**base)


class TestGymCompliance:
    def test_passes_env_checker(self):
        # PRD Feature 7 acceptance: passes gymnasium's env_checker.
        check_env(InventoryEnv(_config(), _episode(), seed=1), skip_render_check=True)

    def test_observation_shape_matches_space(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_action_space_is_eleven(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        assert env.action_space.n == len(ACTION_MULTIPLIERS) == 11


class TestMechanics:
    def test_episode_terminates_after_length(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(0)
            done = term or trunc
            steps += 1
        assert steps == _N

    def test_zero_action_orders_nothing(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        _, _, _, _, info = env.step(0)
        assert info["order"] == 0.0
        assert info["order_cost"] == 0.0

    def test_order_incurs_fixed_plus_purchase_cost(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        # Action index 2 -> multiplier 1.0 -> 10 units at d_bar=10.
        _, _, _, _, info = env.step(2)
        assert info["order"] == pytest.approx(10.0)
        # fixed (10) + purchase (1 * 10) = 20.
        assert info["order_cost"] == pytest.approx(20.0)

    def test_lost_sales_when_demand_exceeds_stock(self):
        # Start stock = forecast_mean[0] = 10; demand 10/day, order nothing.
        # Day 0: meet 10 from 10 -> 0 left. Day 1: demand 10, stock 0 -> all lost.
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        env.step(0)
        _, _, _, _, info = env.step(0)
        assert info["lost_sales"] == pytest.approx(10.0)
        assert info["stockout_cost"] == pytest.approx(2.0 * 10.0)

    def test_reward_is_negative_total_cost(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        _, reward, _, _, info = env.step(2)
        expected = -(info["holding_cost"] + info["stockout_cost"] + info["order_cost"])
        assert reward == pytest.approx(expected)

    def test_orders_arrive_after_lead_time(self):
        # No additional lead-time noise: lead = base = 3. With a zero-demand
        # episode the init stock is forecast_mean[0] = 0, so the day-0 order of
        # 10 units arrives (rolls into on-hand) at the start of day 3's step.
        cfg = _config(lead_time_base=3, lead_time_max_additional=0)
        env = InventoryEnv(cfg, _episode(demand_value=0.0), seed=1)
        env.reset(seed=1)
        assert env.unwrapped._on_hand == pytest.approx(0.0)  # init = forecast_mean[0]
        env.step(2)  # day 0: order 10 units
        env.step(0)  # day 1: nothing arrives
        env.step(0)  # day 2: nothing arrives
        before = env.unwrapped._on_hand
        env.step(0)  # day 3: the order arrives
        after = env.unwrapped._on_hand
        assert before == pytest.approx(0.0)
        assert after == pytest.approx(10.0)


class TestStructuredState:
    def test_current_state_matches_observation_fields(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        env.reset(seed=1)
        s = env.current_state()
        assert s.forecast_mean.shape == (_W,)
        assert s.forecast_std.shape == (_W,)
        assert s.day_of_week.shape == (7,)
        assert s.day_of_week.sum() == 1
        assert s.inventory_position == pytest.approx(s.on_hand)

    def test_order_units_maps_to_nearest_action(self):
        env = InventoryEnv(_config(mean_daily_demand=10.0), _episode(), seed=1)
        # 10 units -> multiplier 1.0 -> action index 2.
        assert env.order_units(10.0) == 2
        # 0 units -> action 0.
        assert env.order_units(0.0) == 0
        # 48 units -> nearest is 50 (mult 5.0) -> index 10.
        assert env.order_units(48.0) == 10


class TestClassicalPoliciesRun:
    def test_eoq_produces_valid_trajectory(self):
        # PRD Feature 7 acceptance: EOQ runs 28 days in the env.
        env = InventoryEnv(_config(), _episode(), seed=42)
        env.reset(seed=42)
        policy = EOQPolicy(holding_per_unit_per_day=0.05, fixed_order_cost=10.0, lead_time=3)
        done = False
        total = 0.0
        while not done:
            action = env.order_units(policy.select_action(env.current_state()))
            _, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        assert np.isfinite(total)

    def test_order_up_to_produces_valid_trajectory(self):
        env = InventoryEnv(_config(), _episode(), seed=42)
        env.reset(seed=42)
        policy = OrderUpToPolicy(lead_time=3, review_period=1)
        done = False
        while not done:
            action = env.order_units(policy.select_action(env.current_state()))
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc
        assert True  # completed without error


class TestValidation:
    def test_rejects_short_demand(self):
        ep = _episode()
        ep.demand = ep.demand[:10]
        with pytest.raises(ValueError, match="demand"):
            InventoryEnv(_config(), ep, seed=1)

    def test_rejects_short_forecast(self):
        ep = _episode()
        ep.forecast_mean = ep.forecast_mean[:_N]  # missing the trailing window
        with pytest.raises(ValueError, match="forecast"):
            InventoryEnv(_config(), ep, seed=1)


class TestDisruptions:
    def test_demand_spike_scales_window_only(self):
        spiked = DemandSpikeWrapper(
            InventoryEnv(_config(), _episode(), seed=1),
            multiplier=1.5,
            start_day=7,
            duration=14,
        )
        d = spiked.unwrapped._episode.demand
        assert d[6] == pytest.approx(10.0)  # before window
        assert d[7] == pytest.approx(15.0)  # window start
        assert d[20] == pytest.approx(15.0)  # window end (inclusive)
        assert d[21] == pytest.approx(10.0)  # after window

    def test_demand_spike_idempotent_across_resets(self):
        spiked = DemandSpikeWrapper(InventoryEnv(_config(), _episode(), seed=1), multiplier=1.5)
        spiked.reset(seed=1)
        spiked.reset(seed=1)
        # Not compounded: still 15, not 22.5.
        assert spiked.unwrapped._episode.demand[7] == pytest.approx(15.0)

    def test_demand_spike_rejects_bad_multiplier(self):
        with pytest.raises(ValueError, match="multiplier"):
            DemandSpikeWrapper(InventoryEnv(_config(), _episode(), seed=1), multiplier=0.0)

    def test_lead_time_disruption_restores_after_step(self):
        env = InventoryEnv(_config(), _episode(), seed=1)
        ltd = LeadTimeDisruptionWrapper(env, multiplier=2.0, start_day=7, duration=14)
        ltd.reset(seed=1)
        for _ in range(_N):
            ltd.step(2)
            # working lead time always restored to base between steps.
            assert env.unwrapped._lead_time_base == 3

    def test_lead_time_disruption_delays_arrivals(self):
        # Order placed inside the window arrives later than base lead time.
        cfg = _config(lead_time_base=3, lead_time_max_additional=3)
        env = InventoryEnv(cfg, _episode(demand_value=0.0), seed=5)
        ltd = LeadTimeDisruptionWrapper(env, multiplier=2.0, start_day=0, duration=5)
        ltd.reset(seed=5)
        before = env.unwrapped._on_hand
        ltd.step(2)  # order during disruption -> lead ~6 days
        # Over the next 3 days (normal lead), nothing should arrive yet.
        for _ in range(3):
            ltd.step(0)
        mid = env.unwrapped._on_hand
        assert mid == pytest.approx(before)  # delayed beyond base lead time

    def test_lead_time_disruption_rejects_multiplier_below_one(self):
        with pytest.raises(ValueError, match="multiplier"):
            LeadTimeDisruptionWrapper(InventoryEnv(_config(), _episode(), seed=1), multiplier=0.5)
