"""Unit tests for the cross-condition resilience helpers.

Covers the replication-averaged daily-service curve and the recovery/degradation
computation edge cases not already exercised in ``test_evaluation.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_scm.evaluation import compute_resilience, mean_daily_service


def _daily(per_rep_lost: list[list[float]], demand: float = 10.0) -> pd.DataFrame:
    """Build a multi-replication daily frame from per-rep lost-sales lists.

    Args:
        per_rep_lost: One list of per-day lost sales per replication.
        demand: Constant daily demand.

    Returns:
        A daily-rows DataFrame.
    """
    rows = []
    for rep, lost_by_day in enumerate(per_rep_lost):
        for day, lost in enumerate(lost_by_day):
            rows.append(
                {
                    "record_type": "daily",
                    "replication": rep,
                    "day": day,
                    "demand": demand,
                    "lost_sales": lost,
                }
            )
    return pd.DataFrame(rows)


class TestMeanDailyService:
    def test_averages_across_replications(self):
        # Rep 0 fully served, rep 1 half lost on every day -> mean service 0.75.
        daily = _daily([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        curve = mean_daily_service(daily)
        assert np.allclose(curve, 0.75)

    def test_length_matches_episode_days(self):
        daily = _daily([[0.0] * 28])
        assert len(mean_daily_service(daily)) == 28

    def test_zero_demand_counts_as_served(self):
        daily = pd.DataFrame(
            [{"record_type": "daily", "replication": 0, "day": 0, "demand": 0.0, "lost_sales": 0.0}]
        )
        assert mean_daily_service(daily)[0] == pytest.approx(1.0)


class TestComputeResilienceEdge:
    def test_window_end_beyond_episode(self):
        # A window end past the episode leaves no post-window days -> recovery 0.
        daily = _daily([[0.0] * 28])
        res = compute_resilience(daily, 1.0, 0.9, window=(7, 40))
        assert res["recovery_time"] == 0.0

    def test_degradation_uses_provided_fills_not_daily(self):
        # Degradation depends only on the scalar fills passed in.
        daily = _daily([[0.0] * 28])
        res = compute_resilience(
            daily, baseline_fill_rate=0.92, disruption_fill_rate=0.6, window=(7, 21)
        )
        assert res["service_level_degradation"] == pytest.approx(0.32)
