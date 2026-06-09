"""ARIMA demand forecaster.

Wraps ``pmdarima.auto_arima`` behind the :class:`~adaptive_scm.forecasting.base.Forecaster`
interface. Order selection is automatic (AIC-minimizing search over non-seasonal
and weekly-seasonal terms); the fitted model produces 28-day point forecasts with
95% confidence intervals and reports a validation-set RMSE used by the simulator
to calibrate demand noise.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from adaptive_scm.data import select_split
from adaptive_scm.forecasting.base import ForecastOutput, Forecaster, rmse
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# 95% confidence interval -> alpha for pmdarima's return_conf_int.
_CI_ALPHA = 0.05


class ARIMAForecaster(Forecaster):
    """Auto-ARIMA forecaster with weekly seasonality.

    Selects ``(p, d, q)(P, D, Q)_m`` orders by minimizing an information
    criterion (AIC by default) via ``pmdarima.auto_arima``, fits on the
    training split only, and forecasts ``horizon`` days ahead in a single call.
    Interchangeable with the XGBoost and TFT forecasters through the
    :class:`Forecaster` interface, so policies and the simulator never depend
    on the concrete type.

    Negative point forecasts and bounds are floored at zero, since the forecast
    feeds an inventory simulator where demand cannot be negative.
    """

    def __init__(
        self,
        seasonal: bool = True,
        seasonal_period: int = 7,
        information_criterion: str = "aic",
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_d: int = 2,
        max_D: int = 1,
        stepwise: bool = True,
    ) -> None:
        """Configure the auto-ARIMA search space.

        Stores the search hyperparameters; no fitting happens here. The first
        three arguments mirror ``config/forecasters/arima.yaml``; the ``max_*``
        bounds and ``stepwise`` flag are additional knobs (defaulting to
        ``auto_arima``'s own defaults) that let tests constrain the search to
        stay fast. This is a documented, additive deviation from the PRD config
        schema, which only specified ``seasonal``/``seasonal_period``/
        ``information_criterion``.

        Args:
            seasonal: Whether to search seasonal terms.
            seasonal_period: Seasonal period ``m`` (7 for weekly seasonality).
            information_criterion: Criterion to minimize (``"aic"``, ``"bic"``, etc.).
            max_p: Maximum non-seasonal AR order.
            max_q: Maximum non-seasonal MA order.
            max_P: Maximum seasonal AR order.
            max_Q: Maximum seasonal MA order.
            max_d: Maximum non-seasonal differencing order.
            max_D: Maximum seasonal differencing order.
            stepwise: Use the stepwise (fast) search instead of full grid.

        Raises:
            ValueError: If ``seasonal_period`` is below 1.
        """
        if seasonal_period < 1:
            raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")

        self._seasonal = bool(seasonal)
        self._seasonal_period = int(seasonal_period)
        self._information_criterion = str(information_criterion)
        self._max_p = int(max_p)
        self._max_q = int(max_q)
        self._max_P = int(max_P)
        self._max_Q = int(max_Q)
        self._max_d = int(max_d)
        self._max_D = int(max_D)
        self._stepwise = bool(stepwise)

        self._model = None  # set by fit; the underlying pmdarima ARIMA
        self._historical_rmse: float | None = None

    @property
    def historical_rmse(self) -> float:
        """Validation-set RMSE recorded during :meth:`fit`.

        Exposed for inspection and logging; the same value is also embedded in
        every :class:`ForecastOutput` this model produces.

        Returns:
            The validation RMSE as a float.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._historical_rmse is None:
            raise RuntimeError("historical_rmse is unavailable until fit() is called")
        return self._historical_rmse

    @property
    def order(self) -> tuple[int, int, int]:
        """Selected non-seasonal ``(p, d, q)`` order.

        Returns:
            The ``(p, d, q)`` tuple chosen by ``auto_arima``.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        self._require_fitted()
        return tuple(self._model.order)

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        """Selected seasonal ``(P, D, Q, m)`` order.

        Returns:
            The ``(P, D, Q, m)`` tuple chosen by ``auto_arima``.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        self._require_fitted()
        return tuple(self._model.seasonal_order)

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit auto-ARIMA on the training split and record validation RMSE.

        Runs ``pmdarima.auto_arima`` over the configured search space on the
        ``sales`` values where ``split == "train"`` (or all rows if no ``split``
        column is present). Then computes ``historical_rmse`` by forecasting the
        ``val`` horizon and comparing against the validation actuals; if no
        validation rows exist, falls back to the in-sample residual RMSE.

        Args:
            train_data: Preprocessed DataFrame with a ``sales`` column and,
                ideally, a ``split`` column labeling ``train``/``val``/``test``.

        Raises:
            ValueError: If ``train_data`` lacks a ``sales`` column or the
                training split is empty.
        """
        if "sales" not in train_data.columns:
            raise ValueError("train_data must contain a 'sales' column")

        if "split" in train_data.columns:
            train_sales = select_split(train_data, "train")["sales"]
            val_sales = select_split(train_data, "val")["sales"]
        else:
            train_sales = train_data["sales"]
            val_sales = pd.Series(dtype=train_data["sales"].dtype)

        train_array = train_sales.to_numpy(dtype=float)
        if train_array.size == 0:
            raise ValueError("training split is empty; nothing to fit")

        import pmdarima as pm

        # auto_arima's default start_p/start_q is 2; if a caller sets a max
        # below that the library errors. Clamp the start to the max so tight
        # search bounds are always valid.
        start_p = min(2, self._max_p)
        start_q = min(2, self._max_q)
        start_P = min(1, self._max_P)
        start_Q = min(1, self._max_Q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = pm.auto_arima(
                train_array,
                seasonal=self._seasonal,
                m=self._seasonal_period,
                information_criterion=self._information_criterion,
                start_p=start_p,
                start_q=start_q,
                start_P=start_P,
                start_Q=start_Q,
                max_p=self._max_p,
                max_q=self._max_q,
                max_P=self._max_P,
                max_Q=self._max_Q,
                max_d=self._max_d,
                max_D=self._max_D,
                stepwise=self._stepwise,
                suppress_warnings=True,
                error_action="ignore",
            )

        self._historical_rmse = self._compute_validation_rmse(val_sales.to_numpy(dtype=float))

        _LOG.info(
            "arima_fitted",
            order=self.order,
            seasonal_order=self.seasonal_order,
            aic=float(self._model.aic()),
            train_days=int(train_array.size),
            val_days=int(val_sales.size),
            historical_rmse=self._historical_rmse,
        )

    def predict(self, horizon: int) -> ForecastOutput:
        """Forecast ``horizon`` days ahead with a 95% confidence interval.

        Calls the fitted model's ``predict(n_periods=horizon, return_conf_int=True)``
        and packages the point forecast and interval into a :class:`ForecastOutput`,
        flooring all values at zero. The interval is the ARIMA predictive 95% CI,
        which plays the role of ``mean ± 1.96σ`` for downstream uncertainty use.

        Args:
            horizon: Number of days to forecast. Must be positive.

        Returns:
            A :class:`ForecastOutput` of length ``horizon`` carrying the recorded
            ``historical_rmse``.

        Raises:
            ValueError: If ``horizon`` is not positive.
            RuntimeError: If called before :meth:`fit`.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self._require_fitted()

        point, conf_int = self._model.predict(
            n_periods=horizon, return_conf_int=True, alpha=_CI_ALPHA
        )
        point = np.clip(np.asarray(point, dtype=float), 0.0, None)
        conf_int = np.asarray(conf_int, dtype=float)
        lower = np.clip(conf_int[:, 0], 0.0, None)
        upper = np.clip(conf_int[:, 1], 0.0, None)

        return ForecastOutput(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            historical_rmse=float(self._historical_rmse),
        )

    def save(self, path: Path) -> None:
        """Persist the fitted model and metadata to ``path`` via joblib.

        Writes a single joblib artifact containing the underlying pmdarima
        model and the recorded ``historical_rmse``, so :meth:`load` restores a
        forecaster that predicts identically without refitting.

        Args:
            path: Destination file path (parent directories are created).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "historical_rmse": self._historical_rmse},
            path,
        )
        _LOG.info("arima_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "ARIMAForecaster":
        """Load a forecaster previously written by :meth:`save`.

        Reconstructs an :class:`ARIMAForecaster`, restoring the pmdarima model
        and ``historical_rmse`` from the joblib artifact. Search hyperparameters
        are not needed post-fit, so the instance is created with defaults and
        then populated.

        Args:
            path: Path passed to a prior :meth:`save` call.

        Returns:
            A fitted :class:`ARIMAForecaster` ready for :meth:`predict`.
        """
        payload = joblib.load(Path(path))
        instance = cls()
        instance._model = payload["model"]
        instance._historical_rmse = float(payload["historical_rmse"])
        _LOG.info("arima_loaded", path=str(path))
        return instance

    def _compute_validation_rmse(self, val_actuals: np.ndarray) -> float:
        """Compute RMSE on the validation horizon, or in-sample as a fallback.

        If validation actuals are provided, forecasts that many steps ahead and
        returns the RMSE against them. With no validation data, returns the RMSE
        of the model's in-sample residuals so ``historical_rmse`` is always
        populated. Called once at the end of :meth:`fit`.

        Args:
            val_actuals: Validation-period sales, possibly length zero.

        Returns:
            RMSE as a non-negative float.
        """
        if val_actuals.size > 0:
            point = np.clip(
                np.asarray(self._model.predict(n_periods=val_actuals.size), dtype=float),
                0.0,
                None,
            )
            return rmse(point, val_actuals)

        resid = np.asarray(self._model.resid(), dtype=float)
        return float(np.sqrt(np.mean(resid**2)))

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted yet.

        Guard shared by :meth:`predict`, :meth:`save`, and the order properties
        so each surfaces a clear error instead of an ``AttributeError`` on
        ``None``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
