"""Unit tests for the hypothesis-testing module (PRD Feature 12).

Builds small synthetic cell Parquets with controlled per-replication costs so
the paired tests have known-sign differences, then checks H1/H2/H3 wiring,
effect-size direction, and Markdown rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_scm.evaluation import (
    paired_comparison,
    per_replication_costs,
    render_hypothesis_markdown,
)
from adaptive_scm.evaluation import test_h1 as run_h1
from adaptive_scm.evaluation import test_h2 as run_h2
from adaptive_scm.evaluation import test_h3 as run_h3


def _write_cell(sim_dir, forecaster, policy, condition, rep_costs):
    """Write a synthetic cell Parquet whose daily rows sum to ``rep_costs``.

    Each replication gets two daily rows whose holding costs sum to that rep's
    target total cost, so :func:`per_replication_costs` recovers ``rep_costs``.

    Args:
        sim_dir: Output directory.
        forecaster: Forecaster label.
        policy: Policy label.
        condition: Condition label.
        rep_costs: Iterable of per-replication total costs.
    """
    rows = []
    for rep, cost in enumerate(rep_costs):
        rows.append(
            {
                "record_type": "daily",
                "replication": rep,
                "day": 0,
                "holding_cost": cost / 2,
                "stockout_cost": 0.0,
                "order_cost": 0.0,
            }
        )
        rows.append(
            {
                "record_type": "daily",
                "replication": rep,
                "day": 1,
                "holding_cost": cost / 2,
                "stockout_cost": 0.0,
                "order_cost": 0.0,
            }
        )
    rows.append(
        {
            "record_type": "summary",
            "replication": np.nan,
            "day": np.nan,
            "holding_cost": np.nan,
            "stockout_cost": np.nan,
            "order_cost": np.nan,
        }
    )
    path = sim_dir / f"{forecaster}_{policy}_{condition}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)


class TestPerReplicationCosts:
    def test_reconstructs_costs(self, tmp_path):
        _write_cell(tmp_path, "arima", "eoq", "baseline", [100.0, 200.0, 300.0])
        df = pd.read_parquet(tmp_path / "arima_eoq_baseline.parquet")
        costs = per_replication_costs(df)
        assert list(costs) == [100.0, 200.0, 300.0]


class TestPairedComparison:
    def test_detects_better_policy(self):
        better = np.array([90.0, 95.0, 88.0, 92.0, 91.0])
        worse = np.array([110.0, 118.0, 105.0, 115.0, 109.0])
        result = paired_comparison(better, worse, "ppo_vs_eoq")
        assert result["favors_better"] is True
        assert result["mean_difference"] < 0  # better has lower cost
        assert result["significant"] is True
        assert result["cohens_d"] < 0

    def test_no_difference_not_significant(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100, 5, 20)
        b = a + rng.normal(0, 0.01, 20)  # essentially identical
        result = paired_comparison(a, b, "x")
        assert result["significant"] is False

    def test_truncates_to_shorter(self):
        result = paired_comparison(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), "x")
        assert result["n"] == 2


class TestH1:
    def test_ppo_vs_classical(self, tmp_path):
        # PPO cheaper than both classical policies for arima/baseline.
        _write_cell(tmp_path, "arima", "ppo", "baseline", [80.0, 82.0, 79.0, 81.0])
        _write_cell(tmp_path, "arima", "eoq", "baseline", [100.0, 102.0, 99.0, 101.0])
        _write_cell(tmp_path, "arima", "order_up_to", "baseline", [110.0, 112.0, 109.0, 111.0])
        rows = run_h1(tmp_path, conditions=("baseline",))
        assert len(rows) == 2  # ppo vs eoq, ppo vs order_up_to
        assert all(r["favors_better"] for r in rows)
        comparisons = {r["comparison"] for r in rows}
        assert comparisons == {"ppo_vs_eoq", "ppo_vs_order_up_to"}

    def test_skips_missing_cells(self, tmp_path):
        # Only PPO present, no classical -> no comparisons.
        _write_cell(tmp_path, "arima", "ppo", "baseline", [80.0, 82.0])
        assert run_h1(tmp_path, conditions=("baseline",)) == []


class TestH2:
    def test_integrated_vs_baselines(self, tmp_path):
        _write_cell(tmp_path, "tft", "ppo", "baseline", [70.0, 72.0, 71.0, 69.0])
        for f in ("arima", "xgboost", "tft"):
            for p in ("eoq", "order_up_to"):
                _write_cell(tmp_path, f, p, "baseline", [100.0, 101.0, 99.0, 100.0])
        rows = run_h2(tmp_path, condition="baseline")
        assert len(rows) == 6  # 3 forecasters x 2 classical policies
        assert all(r["favors_better"] for r in rows)

    def test_no_integrated_cell(self, tmp_path):
        _write_cell(tmp_path, "arima", "eoq", "baseline", [100.0, 101.0])
        assert run_h2(tmp_path, condition="baseline") == []


class TestH3:
    def test_correlation_computed(self):
        suite = pd.DataFrame(
            {
                "forecast_rmse": [1.0, 2.0, 3.0, 4.0, 5.0],
                "total_cost_mean": [100.0, 90.0, 110.0, 95.0, 105.0],
            }
        )
        result = run_h3(suite)
        assert -1.0 <= result["spearman_rho"] <= 1.0
        assert result["n"] == 5

    def test_supports_h3_when_weak(self):
        # Near-zero correlation -> supports H3.
        suite = pd.DataFrame(
            {
                "forecast_rmse": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "total_cost_mean": [100.0, 100.0, 100.0, 100.0, 100.0, 101.0],
            }
        )
        result = run_h3(suite)
        assert result["supports_h3"] is True

    def test_nan_when_constant(self):
        suite = pd.DataFrame({"forecast_rmse": [2.0, 2.0, 2.0], "total_cost_mean": [1.0, 2.0, 3.0]})
        result = run_h3(suite)
        assert np.isnan(result["spearman_rho"])


class TestRendering:
    def test_report_has_all_sections(self, tmp_path):
        _write_cell(tmp_path, "arima", "ppo", "baseline", [80.0, 82.0, 79.0, 81.0])
        _write_cell(tmp_path, "arima", "eoq", "baseline", [100.0, 102.0, 99.0, 101.0])
        _write_cell(tmp_path, "arima", "order_up_to", "baseline", [110.0, 112.0, 109.0, 111.0])
        h1 = run_h1(tmp_path, conditions=("baseline",))
        suite = pd.DataFrame(
            {"forecast_rmse": [1.0, 2.0, 3.0], "total_cost_mean": [90.0, 100.0, 110.0]}
        )
        md = render_hypothesis_markdown(h1, [], run_h3(suite))
        assert "# Hypothesis Tests" in md
        assert "## H1" in md
        assert "## H2" in md
        assert "## H3" in md
        assert "order-up-to" in md  # load-bearing comparison called out
