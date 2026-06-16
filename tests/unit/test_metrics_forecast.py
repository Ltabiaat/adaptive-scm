"""Unit tests for the forecast-accuracy helpers (RMSE and MAPE, Section 3.7).

Pure numeric helpers with no heavy dependencies, so they run in the fast
(non-torch) test group.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_scm.evaluation import forecast_accuracy_table
from adaptive_scm.forecasting import forecast_accuracy, mape, rmse


class TestRMSE:
    def test_basic(self):
        assert rmse(np.array([12.0, 18.0]), np.array([10.0, 20.0])) == pytest.approx(2.0)

    def test_zero_when_perfect(self):
        assert rmse(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == pytest.approx(0.0)


class TestMAPE:
    def test_excludes_zero_demand_days(self):
        # actual=[10,20,0,5], pred=[12,18,3,5]: nonzero-day errors 0.2, 0.1, 0.0.
        m = mape(np.array([12.0, 18.0, 3.0, 5.0]), np.array([10.0, 20.0, 0.0, 5.0]))
        assert m == pytest.approx(10.0)

    def test_perfect_forecast_is_zero(self):
        assert mape(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == pytest.approx(0.0)

    def test_all_zero_actual_is_nan(self):
        assert np.isnan(mape(np.array([1.0, 2.0]), np.array([0.0, 0.0])))

    def test_is_a_percentage(self):
        # 50% under-forecast on every day -> MAPE 50.
        actual = np.array([10.0, 20.0, 30.0])
        m = mape(actual * 0.5, actual)
        assert m == pytest.approx(50.0)


class TestForecastAccuracy:
    def test_returns_both_metrics(self):
        acc = forecast_accuracy(np.array([12.0, 18.0]), np.array([10.0, 20.0]))
        assert set(acc) == {"rmse", "mape"}
        assert acc["rmse"] == pytest.approx(2.0)

    def test_accuracy_table_ranks_forecasters(self):
        actual = np.array([10.0, 20.0, 5.0])
        table = forecast_accuracy_table(
            {"arima": np.array([12.0, 22.0, 8.0]), "tft": actual.copy()}, actual
        )
        assert list(table.columns) == ["rmse", "mape"]
        # TFT here is a perfect forecast, so it has the lower error on both.
        assert table.loc["tft", "rmse"] < table.loc["arima", "rmse"]
        assert table.loc["tft", "mape"] < table.loc["arima", "mape"]
