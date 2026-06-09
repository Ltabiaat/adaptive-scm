"""XGBoost demand forecaster.

Implements the :class:`~adaptive_scm.forecasting.base.Forecaster` interface with
gradient-boosted trees over the full engineered feature set. Order selection is
a grid search with early stopping on one-step validation RMSE; the 28-day
forecast is produced recursively (each day's prediction feeds the next day's
lag and rolling features). See design decisions D-3.1 .. D-3.6.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from adaptive_scm.data import feature_columns, select_split
from adaptive_scm.data.features import LAGS, ROLLING_WINDOWS
from adaptive_scm.forecasting.base import ForecastOutput, Forecaster, rmse
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# 95% interval half-width multiplier (D-3.5).
_Z_95 = 1.96

# Default grid (PRD Feature 3). Overridable via config / constructor.
_DEFAULT_GRID = {
    "max_depth": (3, 6, 9),
    "learning_rate": (0.01, 0.05, 0.1),
    "n_estimators": (100, 300, 500),
    "reg_lambda": (0.0, 0.1, 1.0),
}


class XGBoostForecaster(Forecaster):
    """Gradient-boosted-tree forecaster with recursive multi-step prediction.

    Grid-searches ``max_depth × learning_rate × n_estimators × reg_lambda`` with
    early stopping on one-step validation RMSE, keeps the best model, then
    forecasts recursively: predicted sales feed back as lag/rolling inputs for
    later horizon steps while known future covariates (calendar, event, price)
    are read from rows stashed during :meth:`fit`. Interchangeable with the
    ARIMA and TFT forecasters via the :class:`Forecaster` interface.
    """

    def __init__(
        self,
        grid: dict[str, tuple] | None = None,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        """Configure the grid search.

        Stores the hyperparameter grid and early-stopping patience; no training
        happens here. The grid mirrors ``config/forecasters/xgboost.yaml``;
        tests pass a small grid to stay fast.

        Args:
            grid: Mapping of XGBoost param name to a tuple of candidate values.
                Defaults to the PRD's 3×3×3×3 grid.
            early_stopping_rounds: Patience (rounds without val-RMSE improvement).
            random_state: Seed passed to every XGBoost model for reproducibility.

        Raises:
            ValueError: If ``early_stopping_rounds`` is not positive.
        """
        if early_stopping_rounds < 1:
            raise ValueError(f"early_stopping_rounds must be >= 1, got {early_stopping_rounds}")
        self._grid = dict(grid) if grid is not None else dict(_DEFAULT_GRID)
        self._early_stopping_rounds = int(early_stopping_rounds)
        self._random_state = int(random_state)

        self._model = None  # best fitted xgboost.XGBRegressor
        self._best_params: dict | None = None
        self._feature_columns: list[str] | None = None
        self._future_features: pd.DataFrame | None = None  # post-train rows (engineered)
        self._train_sales: np.ndarray | None = None  # buffer seed for recursion
        self._historical_rmse: float | None = None

    @property
    def historical_rmse(self) -> float:
        """Recursive validation RMSE recorded during :meth:`fit` (see D-3.4).

        Returns:
            The recursive multi-step validation RMSE.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._historical_rmse is None:
            raise RuntimeError("historical_rmse is unavailable until fit() is called")
        return self._historical_rmse

    @property
    def best_params(self) -> dict:
        """Hyperparameters of the grid-search winner.

        Returns:
            The best parameter dict (includes the early-stopped ``n_estimators``).

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        self._require_fitted()
        return dict(self._best_params)

    def fit(self, train_data: pd.DataFrame) -> None:
        """Grid-search XGBoost on the train/val splits and store recursion state.

        Trains every grid point on the ``train`` split with early stopping on
        the ``val`` split (one-step RMSE), selects the lowest one-step val RMSE,
        then computes ``historical_rmse`` as the selected model's recursive
        multi-step RMSE over the val horizon. Also stashes the engineered
        post-train rows and the training-sales buffer that :meth:`predict` needs
        for recursive forecasting.

        Args:
            train_data: Engineered, split-labeled frame (see :func:`feature_columns`).

        Raises:
            ValueError: If required columns or splits are missing.
        """
        if "sales" not in train_data.columns or "split" not in train_data.columns:
            raise ValueError("train_data must contain 'sales' and 'split' columns")

        train_df = select_split(train_data, "train")
        val_df = select_split(train_data, "val")
        if train_df.empty or val_df.empty:
            raise ValueError("both 'train' and 'val' splits must be non-empty")

        self._feature_columns = feature_columns(train_data)
        x_train = train_df[self._feature_columns].to_numpy(dtype=float)
        y_train = train_df["sales"].to_numpy(dtype=float)
        x_val = val_df[self._feature_columns].to_numpy(dtype=float)
        y_val = val_df["sales"].to_numpy(dtype=float)

        # Future covariate rows (everything after train) + recursion buffer seed.
        self._future_features = train_data.loc[~train_data.index.isin(train_df.index)].copy()
        self._train_sales = y_train.copy()

        best = self._grid_search(x_train, y_train, x_val, y_val)
        self._model = best["model"]
        self._best_params = best["params"]

        # historical_rmse: recursive multi-step error over the val horizon (D-3.4).
        recursive_val = self._recursive_forecast(len(val_df))
        self._historical_rmse = rmse(recursive_val, y_val)

        _LOG.info(
            "xgboost_fitted",
            best_params=self._best_params,
            one_step_val_rmse=best["one_step_rmse"],
            recursive_val_rmse=self._historical_rmse,
            n_features=len(self._feature_columns),
            grid_points=best["n_evaluated"],
        )

    def predict(self, horizon: int) -> ForecastOutput:
        """Recursively forecast ``horizon`` days ahead with approximate CIs.

        Builds the forecast one day at a time (D-3.2), reading known future
        covariates from the stashed post-train rows and recomputing sales-derived
        lag/rolling features from the growing prediction buffer. Confidence
        bounds are ``point ± 1.96 · historical_rmse`` (D-3.5). All values are
        floored at zero (D-2.2).

        Args:
            horizon: Number of days to forecast. Must be positive and not exceed
                the number of stashed future rows.

        Returns:
            A :class:`ForecastOutput` of length ``horizon``.

        Raises:
            ValueError: If ``horizon`` is out of range.
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon > len(self._future_features):
            raise ValueError(
                f"horizon {horizon} exceeds available future rows "
                f"({len(self._future_features)})"
            )

        point = self._recursive_forecast(horizon)
        margin = _Z_95 * self._historical_rmse
        lower = np.clip(point - margin, 0.0, None)
        upper = np.clip(point + margin, 0.0, None)
        return ForecastOutput(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            historical_rmse=float(self._historical_rmse),
        )

    def save(self, path: Path) -> None:
        """Persist the model and all recursion state via joblib.

        Bundles the fitted model, best params, feature column order, stashed
        future rows, training-sales buffer, and ``historical_rmse`` so
        :meth:`load` restores a forecaster that predicts identically.

        Args:
            path: Destination file path (parent directories are created).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "best_params": self._best_params,
                "feature_columns": self._feature_columns,
                "future_features": self._future_features,
                "train_sales": self._train_sales,
                "historical_rmse": self._historical_rmse,
            },
            path,
        )
        _LOG.info("xgboost_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "XGBoostForecaster":
        """Load a forecaster previously written by :meth:`save`.

        Args:
            path: Path passed to a prior :meth:`save` call.

        Returns:
            A fitted :class:`XGBoostForecaster` ready for :meth:`predict`.
        """
        payload = joblib.load(Path(path))
        instance = cls()
        instance._model = payload["model"]
        instance._best_params = payload["best_params"]
        instance._feature_columns = payload["feature_columns"]
        instance._future_features = payload["future_features"]
        instance._train_sales = payload["train_sales"]
        instance._historical_rmse = float(payload["historical_rmse"])
        _LOG.info("xgboost_loaded", path=str(path))
        return instance

    def _grid_search(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Train every grid point and return the lowest one-step-val-RMSE model.

        Each candidate trains with early stopping on the validation eval set
        (so its grid ``n_estimators`` acts as a cap and the early-stopped count
        is recorded). Selection is by one-step validation RMSE (D-3.4).

        Args:
            x_train: Training feature matrix.
            y_train: Training targets.
            x_val: Validation feature matrix (one-step features).
            y_val: Validation targets.

        Returns:
            Dict with keys ``model``, ``params``, ``one_step_rmse``, ``n_evaluated``.
        """
        import xgboost as xgb

        keys = list(self._grid.keys())
        best: dict | None = None
        n_evaluated = 0

        for combo in itertools.product(*(self._grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=self._early_stopping_rounds,
                random_state=self._random_state,
                verbosity=0,
                **params,
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            one_step_rmse = rmse(model.predict(x_val), y_val)
            n_evaluated += 1

            if best is None or one_step_rmse < best["one_step_rmse"]:
                resolved = dict(params)
                resolved["n_estimators"] = int(model.best_iteration) + 1
                best = {
                    "model": model,
                    "params": resolved,
                    "one_step_rmse": one_step_rmse,
                }

        best["n_evaluated"] = n_evaluated
        return best

    def _recursive_forecast(self, horizon: int) -> np.ndarray:
        """Produce a ``horizon``-day recursive point forecast.

        Seeds a sales buffer with the training-sales array, then for each future
        day reads the stashed covariate row, overwrites its lag/rolling features
        from the buffer (rolling std uses ``ddof=1`` to match pandas), predicts,
        floors at zero, and appends the prediction to the buffer (D-3.2).

        Args:
            horizon: Number of days to forecast (assumed within range by callers).

        Returns:
            Point forecast array of shape ``(horizon,)``.
        """
        buffer = list(self._train_sales)
        future = self._future_features.iloc[:horizon].reset_index(drop=True)
        preds = np.empty(horizon, dtype=float)

        for h in range(horizon):
            row = future.loc[h, self._feature_columns].to_dict()
            for k in LAGS:
                row[f"sales_lag_{k}"] = buffer[-k]
            for w in ROLLING_WINDOWS:
                window = np.asarray(buffer[-w:], dtype=float)
                row[f"sales_roll_mean_{w}"] = float(window.mean())
                row[f"sales_roll_std_{w}"] = float(window.std(ddof=1))

            x = np.array([row[c] for c in self._feature_columns], dtype=float).reshape(1, -1)
            yhat = max(0.0, float(self._model.predict(x)[0]))
            preds[h] = yhat
            buffer.append(yhat)

        return preds

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted yet.

        Guard shared by :meth:`predict`, :meth:`save`, and the read-only
        properties.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
