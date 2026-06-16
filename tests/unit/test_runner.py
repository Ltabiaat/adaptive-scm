"""Unit tests for the multi-replication runner.

Drives the real environment with classical policies (no trained artifacts
needed) so the runner's per-day recording, metric wiring, replication seeding,
and persistable-DataFrame layout are all exercised quickly.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_scm.policies import EOQPolicy, OrderUpToPolicy
from adaptive_scm.simulation import (
    DemandSpikeWrapper,
    EnvConfig,
    InventoryEnv,
    build_eval_episode,
    result_to_dataframe,
    run_replications,
)

_EPISODE_LEN = 28


def _series(length: int = 200):
    rng = np.random.default_rng(0)
    days = np.arange(length)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 1, length), 1, None)
    return sales, (days % 7)


def _env(episode_len: int = _EPISODE_LEN, seed: int = 0):
    sales, dow = _series()
    d_bar = float(sales.mean())
    cfg = EnvConfig(episode_length=episode_len, mean_daily_demand=d_bar, demand_noise_cv=0.25)
    # Tier-2: prediction (policy's view) and realized truth are separate; here a
    # near-perfect synthetic prediction over the last window suffices for tests.
    prediction = sales[-episode_len:]
    realized = sales[-episode_len:]
    episode = build_eval_episode(prediction, realized, dow[-episode_len:], 2.0, episode_len)
    return InventoryEnv(cfg, episode, seed=seed)


def _eoq():
    return EOQPolicy(holding_per_unit_per_day=0.05, fixed_order_cost=10.0, lead_time=3)


class TestRunReplications:
    def test_produces_summary_and_daily_rows(self):
        result = run_replications(_env(), _eoq(), n_replications=4)
        # 4 reps x 28 days of daily rows.
        assert len(result.daily) == 4 * _EPISODE_LEN
        assert len(result.per_replication) == 4
        assert "total_cost_mean" in result.summary
        assert "total_cost_std" in result.summary

    def test_daily_has_required_columns(self):
        result = run_replications(_env(), _eoq(), n_replications=2)
        for col in ("replication", "day", "demand", "order", "on_hand", "reward", "sales"):
            assert col in result.daily.columns

    def test_seeds_give_reproducible_results(self):
        a = run_replications(_env(seed=1), _eoq(), n_replications=3, seeds=[10, 11, 12])
        b = run_replications(_env(seed=1), _eoq(), n_replications=3, seeds=[10, 11, 12])
        assert a.summary["total_cost_mean"] == pytest.approx(b.summary["total_cost_mean"])

    def test_different_seeds_differ(self):
        # Stochastic demand: different seeds should generally differ.
        a = run_replications(_env(seed=1), _eoq(), n_replications=3, seeds=[1, 2, 3])
        b = run_replications(_env(seed=1), _eoq(), n_replications=3, seeds=[4, 5, 6])
        assert a.summary["total_cost_mean"] != b.summary["total_cost_mean"]

    def test_rejects_bad_replication_count(self):
        with pytest.raises(ValueError, match="n_replications"):
            run_replications(_env(), _eoq(), n_replications=0)

    def test_rejects_mismatched_seeds(self):
        with pytest.raises(ValueError, match="seeds"):
            run_replications(_env(), _eoq(), n_replications=3, seeds=[1, 2])

    def test_works_with_order_up_to(self):
        policy = OrderUpToPolicy(lead_time=3, review_period=1)
        result = run_replications(_env(), policy, n_replications=2)
        assert result.summary["fill_rate_mean"] >= 0.0

    def test_single_replication_under_30s(self):
        import time

        t0 = time.time()
        run_replications(_env(), _eoq(), n_replications=1)
        assert time.time() - t0 < 30.0  # PRD Feature 10 acceptance


class TestDisruptionRun:
    def test_disruption_run_completes_and_lowers_fill(self):
        # A demand spike should reduce fill rate vs baseline; resilience metrics
        # are computed cross-condition at the suite level, not in this summary.
        baseline = run_replications(_env(seed=1), _eoq(), n_replications=3, seeds=[1, 2, 3])
        spiked_env = DemandSpikeWrapper(_env(seed=1), multiplier=1.5, start_day=7, duration=14)
        spiked = run_replications(spiked_env, _eoq(), n_replications=3, seeds=[1, 2, 3])
        assert "service_level_degradation_mean" not in spiked.summary
        assert spiked.summary["fill_rate_mean"] <= baseline.summary["fill_rate_mean"] + 1e-9


class TestResultToDataFrame:
    def test_layout_has_daily_and_summary(self):
        result = run_replications(_env(), _eoq(), n_replications=2)
        df = result_to_dataframe(result)
        assert (df["record_type"] == "daily").sum() == 2 * _EPISODE_LEN
        assert (df["record_type"] == "summary").sum() == 1
