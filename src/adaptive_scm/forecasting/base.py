"""Abstract base class and output container for forecasting models.

Defines the `Forecaster` interface that ARIMA, XGBoost, and TFT all
implement, plus the `ForecastOutput` dataclass returned by `predict`.
This module is imported by every concrete forecaster and by the
simulation environment (for forecast features in PPO state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastOutput:
    """Container for forecaster output over a fixed horizon.

    Carries both point predictions and uncertainty bounds in a uniform
    shape so downstream policies (order-up-to, PPO) can consume any
    forecaster's output identically.

    Attributes:
        point_forecast: Point predictions, shape (horizon,). For TFT this
            is the P50 quantile; for ARIMA/XGBoost this is the mean prediction.
        lower_bound: Lower uncertainty bound, shape (horizon,) or None. P10
            for TFT; mean - 1.96 * sigma for ARIMA/XGBoost.
        upper_bound: Upper uncertainty bound, shape (horizon,) or None. P90
            for TFT; mean + 1.96 * sigma for ARIMA/XGBoost.
        historical_rmse: RMSE on the validation split. Used by the simulation
            to calibrate lognormal demand noise (see PRD Feature 7).
    """

    point_forecast: np.ndarray
    lower_bound: np.ndarray | None
    upper_bound: np.ndarray | None
    historical_rmse: float

    def __post_init__(self) -> None:
        """Validate array shapes are consistent across point and bounds.

        Performs a simple shape check to catch wiring mistakes early.
        Called automatically by the dataclass after __init__.
        """
        horizon = self.point_forecast.shape[0]
        if self.lower_bound is not None and self.lower_bound.shape[0] != horizon:
            raise ValueError(
                f"lower_bound shape {self.lower_bound.shape} does not match "
                f"point_forecast horizon {horizon}"
            )
        if self.upper_bound is not None and self.upper_bound.shape[0] != horizon:
            raise ValueError(
                f"upper_bound shape {self.upper_bound.shape} does not match "
                f"point_forecast horizon {horizon}"
            )
        if self.historical_rmse < 0:
            raise ValueError(f"historical_rmse must be non-negative, got {self.historical_rmse}")


class Forecaster(ABC):
    """Abstract interface for all demand forecasting models.

    Every concrete forecaster (ARIMA, XGBoost, TFT) implements this four-method
    contract. The simulation runner and experiment scripts depend only on this
    interface, never on concrete classes, which is what makes forecasters
    interchangeable in the experiment grid.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the model on historical sales data.

        Called once per (forecaster, product-store) combination during the
        offline training phase before any simulation runs.

        Args:
            train_data: Time-ordered DataFrame containing at minimum a `sales`
                column and a `date` index/column. Additional columns (engineered
                features) are used by ML/DL forecasters and ignored by ARIMA.
        """

    @abstractmethod
    def predict(self, horizon: int) -> ForecastOutput:
        """Produce a forecast for the next `horizon` days.

        For ARIMA and TFT this is a single-shot prediction; for XGBoost it is
        recursive (predict t, feed back as lag for t+1, etc.).

        Args:
            horizon: Number of days to forecast (typically 28 per PRD).

        Returns:
            ForecastOutput containing point predictions, uncertainty bounds,
            and historical RMSE.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk.

        Saves all state needed to reproduce predictions after `load`. Called
        once per trained forecaster; the simulation loads frozen forecasters
        rather than retraining mid-experiment.

        Args:
            path: Destination path. Extension is forecaster-specific.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Forecaster:
        """Restore a previously saved model.

        Inverse of `save`. Must round-trip cleanly: a loaded forecaster
        produces identical predictions to the original.

        Args:
            path: Path to the saved model artifact.

        Returns:
            A ready-to-predict Forecaster instance.
        """
