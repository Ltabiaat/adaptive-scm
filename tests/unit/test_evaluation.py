"""Unit tests for evaluation metrics and the suite analyzer.

Tests metric computation on hand-built trajectories (so expected values are
exact) and the analyzer's aggregation / correlation / Markdown rendering on a
small synthetic suite table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_scm.evaluation import (
    aggregate_metrics,
    collect_summary_rows,
    compute_episode_metrics,
    render_summary_markdown,
    rmse_cost_correlation,
)


def _record(demand, lost, holding=0.0, stockout=0.0, order=0.0):
    """Build one per-day record dict.

    Args:
        demand: Day's demand.
        lost: Day's lost sales.
        holding: Holding cost.
        stockout: Stockout cost.
        order: Order cost.

    Returns:
        A record dict in the shape the runner emits.
    """
    return {
        "demand": demand,
        "lost_sales": lost,
        "holding_cost": holding,
        "stockout_cost": stockout,
        "order_cost": order,
    }


class TestEpisodeMetrics:
    def test_costs_sum_correctly(self):
        records = [
            _record(10, 0, holding=1.0, order=20.0),
            _record(10, 0, holding=2.0, stockout=0.0),
        ]
        m = compute_episode_metrics(records)
        assert m["total_cost"] == pytest.approx(23.0)
        assert m["holding_cost"] == pytest.approx(3.0)
        assert m["order_cost"] == pytest.approx(20.0)

    def test_perfect_fill_rate(self):
        records = [_record(10, 0), _record(8, 0)]
        m = compute_episode_metrics(records)
        assert m["fill_rate"] == pytest.approx(1.0)
        assert m["stockout_frequency"] == pytest.approx(0.0)

    def test_fill_rate_with_lost_sales(self):
        # 5 lost out of 20 total demand -> 75% fill.
        records = [_record(10, 5, stockout=10.0), _record(10, 0)]
        m = compute_episode_metrics(records)
        assert m["fill_rate"] == pytest.approx(0.75)
        assert m["stockout_frequency"] == pytest.approx(0.5)  # 1 of 2 days

    def test_zero_demand_is_full_service(self):
        records = [_record(0, 0)]
        m = compute_episode_metrics(records)
        assert m["fill_rate"] == pytest.approx(1.0)

    def test_baseline_has_no_resilience_penalty(self):
        records = [_record(10, 2, stockout=4.0) for _ in range(28)]
        m = compute_episode_metrics(records, disruption_window=None)
        assert m["service_level_degradation"] == 0.0
        assert m["recovery_time"] == 0.0


class TestResilience:
    def test_degradation_detected_in_window(self):
        # Full service before/after, heavy loss inside the window.
        records = []
        for t in range(28):
            if 7 <= t < 21:
                records.append(_record(10, 8))  # 20% service inside window
            else:
                records.append(_record(10, 0))  # full service outside
        m = compute_episode_metrics(records, disruption_window=(7, 21))
        # pre-window service ~1.0, in-window ~0.2 -> degradation ~0.8.
        assert m["service_level_degradation"] == pytest.approx(0.8, abs=0.05)

    def test_recovery_time_zero_when_immediate(self):
        # Service fully restored the day the window ends.
        records = []
        for t in range(28):
            records.append(_record(10, 8 if 7 <= t < 21 else 0))
        m = compute_episode_metrics(records, disruption_window=(7, 21))
        assert m["recovery_time"] == pytest.approx(0.0, abs=1.0)

    def test_no_recovery_returns_remaining_days(self):
        # Service never recovers after the window -> recovery = remaining days.
        records = []
        for t in range(28):
            records.append(_record(10, 0 if t < 7 else 8))
        m = compute_episode_metrics(records, disruption_window=(7, 21))
        assert m["recovery_time"] == pytest.approx(7.0, abs=1.0)  # days 21..27


class TestAggregate:
    def test_means_and_total_cost_std(self):
        per_rep = [
            {
                "total_cost": 100.0,
                "fill_rate": 0.9,
                "recovery_time": 0.0,
                "holding_cost": 0.0,
                "stockout_cost": 0.0,
                "order_cost": 0.0,
                "stockout_frequency": 0.0,
                "service_level_degradation": 0.0,
            },
            {
                "total_cost": 200.0,
                "fill_rate": 0.8,
                "recovery_time": 0.0,
                "holding_cost": 0.0,
                "stockout_cost": 0.0,
                "order_cost": 0.0,
                "stockout_frequency": 0.0,
                "service_level_degradation": 0.0,
            },
        ]
        agg = aggregate_metrics(per_rep)
        assert agg["total_cost_mean"] == pytest.approx(150.0)
        assert agg["total_cost_std"] == pytest.approx(50.0)
        assert agg["fill_rate_mean"] == pytest.approx(0.85)
        assert agg["n_replications"] == 2.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no replications"):
            aggregate_metrics([])


class TestAnalyzer:
    def _suite(self) -> pd.DataFrame:
        rows = []
        for i, (f, p, c) in enumerate(
            [
                ("arima", "eoq", "baseline"),
                ("arima", "ppo", "baseline"),
                ("xgboost", "eoq", "baseline"),
                ("tft", "ppo", "demand_spike"),
            ]
        ):
            rows.append(
                {
                    "forecaster": f,
                    "policy": p,
                    "condition": c,
                    "total_cost_mean": 100.0 + 10 * i,
                    "fill_rate_mean": 0.9 - 0.02 * i,
                    "service_level_degradation_mean": 0.1 * i,
                    "recovery_time_mean": float(i),
                    "forecast_rmse": 2.0 + 0.5 * i,
                }
            )
        return collect_summary_rows(rows)

    def test_collect_sorts_by_labels(self):
        suite = self._suite()
        assert list(suite.columns[:3]) == ["forecaster", "policy", "condition"]
        assert len(suite) == 4

    def test_rmse_cost_correlation_runs(self):
        rho, p = rmse_cost_correlation(self._suite())
        assert -1.0 <= rho <= 1.0
        assert 0.0 <= p <= 1.0

    def test_correlation_nan_when_too_few(self):
        rho, p = rmse_cost_correlation(self._suite().iloc[:2])
        assert np.isnan(rho)

    def test_markdown_has_expected_sections(self):
        md = render_summary_markdown(self._suite())
        assert "# Experimental Suite Summary" in md
        assert "Total cost" in md
        assert "Fill rate" in md
        assert "H3" in md
