"""Abstract forecaster interface and shared ``ForecastOutput`` dataclass.

Defines the contract every concrete forecaster (ARIMA, XGBoost, TFT) must
satisfy: ``fit``, ``predict``, ``save``, ``load``. Policies (EOQ, OrderUpTo, PPO)
consume ``ForecastOutput`` instances, never concrete forecaster types, so the
forecasting and policy layers stay decoupled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root-mean-square error between two equal-length arrays.

    Shared by all forecasters to compute validation RMSE consistently. Both
    inputs are cast to float; no NaN handling is performed (callers pass clean
    arrays).

    Args:
        predicted: Forecast values.
        actual: Ground-truth values, same shape as ``predicted``.

    Returns:
        RMSE as a non-negative float.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute percentage error, excluding zero-demand days.

    Computes ``(100/n) * sum(|actual - predicted| / actual)`` over the days where
    ``actual`` is non-zero (Section 3.7: MAPE is undefined when ``y_t = 0`` and
    those days are excluded). Returns ``nan`` if every day has zero demand.

    Args:
        predicted: Forecast values.
        actual: Ground-truth values, same shape as ``predicted``.

    Returns:
        MAPE as a percentage (non-negative float), or ``nan`` if undefined.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    nonzero = actual != 0.0
    if not nonzero.any():
        return float("nan")
    errors = np.abs(actual[nonzero] - predicted[nonzero]) / np.abs(actual[nonzero])
    return float(100.0 * errors.mean())


def forecast_accuracy(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return both forecast-accuracy metrics (RMSE and MAPE) for a prediction.

    Convenience wrapper used for reporting a forecaster's test-set accuracy in a
    single call (Section 3.7 reports both). RMSE penalizes large errors and is
    most relevant to inventory cost; MAPE is a scale-independent complement.

    Args:
        predicted: Forecast values.
        actual: Ground-truth values, same shape as ``predicted``.

    Returns:
        Dict with ``rmse`` and ``mape``.
    """
    return {"rmse": rmse(predicted, actual), "mape": mape(predicted, actual)}


@dataclass(frozen=True)
class ForecastOutput:
    """Container for a single multi-step forecast.

    Holds the point forecast plus optional lower and upper quantile bounds and
    the historical RMSE of the source model. Produced by ``Forecaster.predict``
    and consumed by every policy and by the simulation environment (to
    calibrate demand noise). Frozen so it can be hashed and safely passed
    between processes.

    Attributes:
        point_forecast: Mean (ARIMA/XGBoost) or P50 (TFT) forecast of shape ``(horizon,)``.
        lower_bound: P10 (TFT) or ``mean - 1.96 * sigma`` (ARIMA/XGBoost). Optional.
        upper_bound: P90 (TFT) or ``mean + 1.96 * sigma`` (ARIMA/XGBoost). Optional.
        historical_rmse: Validation RMSE; used by the simulator for lognormal demand noise.
    """

    point_forecast: np.ndarray
    lower_bound: np.ndarray | None
    upper_bound: np.ndarray | None
    historical_rmse: float

    def __post_init__(self) -> None:
        """Validate shape and dtype invariants.

        Verifies ``point_forecast`` is 1-D, ``historical_rmse`` is non-negative,
        and that bounds (when present) match the point forecast's shape. Raised
        errors surface contract violations from forecaster implementations as
        soon as a forecast is produced rather than later in the simulator.
        """
        if self.point_forecast.ndim != 1:
            raise ValueError(f"point_forecast must be 1-D, got shape {self.point_forecast.shape}")
        if self.historical_rmse < 0:
            raise ValueError(f"historical_rmse must be non-negative, got {self.historical_rmse}")
        for name, bound in (("lower_bound", self.lower_bound), ("upper_bound", self.upper_bound)):
            if bound is None:
                continue
            if bound.shape != self.point_forecast.shape:
                raise ValueError(
                    f"{name} shape {bound.shape} does not match "
                    f"point_forecast shape {self.point_forecast.shape}"
                )

    @property
    def horizon(self) -> int:
        """Number of forecasted steps.

        Returns:
            Length of ``point_forecast``.
        """
        return int(self.point_forecast.shape[0])


class Forecaster(ABC):
    """Abstract base class for all forecasters.

    Defines the four-method contract (``fit``, ``predict``, ``save``, ``load``)
    that ARIMA, XGBoost, and TFT implementations must satisfy. Concrete classes
    are interchangeable from the perspective of policies, the simulator, and
    the experiment runner.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the model on historical data.

        Implementations are free to use whichever columns from ``train_data``
        they need (e.g. ARIMA uses only ``sales``; XGBoost uses the full feature
        set). The forecaster is expected to retain enough internal state after
        ``fit`` to produce forecasts via ``predict`` without further input.

        Args:
            train_data: DataFrame with at minimum a ``date`` index and a ``sales`` column.
        """

    @abstractmethod
    def predict(self, horizon: int) -> ForecastOutput:
        """Produce a ``horizon``-day forecast starting from the end of training.

        Args:
            horizon: Number of forecast steps. Must be positive.

        Returns:
            A ``ForecastOutput`` of length ``horizon``.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to ``path``.

        Implementations should write a single artifact (e.g. a pickle, a JSON +
        weights bundle, or a directory) so that ``load`` can round-trip it.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster":
        """Load a previously saved forecaster from ``path``.

        Args:
            path: Same path passed to a prior ``save`` call.

        Returns:
            A forecaster ready to call ``predict``.
        """
