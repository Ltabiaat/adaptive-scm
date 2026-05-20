"""Tests for the ForecastOutput dataclass shape and value validation."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_scm.forecasting.base import ForecastOutput


def test_forecast_output_accepts_consistent_shapes() -> None:
    horizon = 28
    fo = ForecastOutput(
        point_forecast=np.zeros(horizon),
        lower_bound=np.zeros(horizon),
        upper_bound=np.zeros(horizon),
        historical_rmse=1.0,
    )
    assert fo.point_forecast.shape == (horizon,)


def test_forecast_output_allows_none_bounds() -> None:
    fo = ForecastOutput(
        point_forecast=np.arange(28, dtype=float),
        lower_bound=None,
        upper_bound=None,
        historical_rmse=0.5,
    )
    assert fo.lower_bound is None
    assert fo.upper_bound is None


def test_forecast_output_rejects_mismatched_lower() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        ForecastOutput(
            point_forecast=np.zeros(28),
            lower_bound=np.zeros(27),
            upper_bound=None,
            historical_rmse=1.0,
        )


def test_forecast_output_rejects_mismatched_upper() -> None:
    with pytest.raises(ValueError, match="upper_bound"):
        ForecastOutput(
            point_forecast=np.zeros(28),
            lower_bound=None,
            upper_bound=np.zeros(29),
            historical_rmse=1.0,
        )


def test_forecast_output_rejects_negative_rmse() -> None:
    with pytest.raises(ValueError, match="historical_rmse"):
        ForecastOutput(
            point_forecast=np.zeros(28),
            lower_bound=None,
            upper_bound=None,
            historical_rmse=-0.1,
        )
