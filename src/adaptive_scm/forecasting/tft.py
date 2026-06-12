"""Temporal Fusion Transformer demand forecaster.

Wraps PyTorch Forecasting's reference TFT behind the
:class:`~adaptive_scm.forecasting.base.Forecaster` interface. Inputs are
partitioned per PRD Feature 4 (static categoricals; time-varying known
calendar/event/price covariates; sales as the time-varying unknown target),
training uses quantile loss over P10/P50/P90 with early stopping on validation
loss, and the 28-day probabilistic forecast is produced in a single forward
pass (not recursively). See design decisions D-4.1 .. D-4.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from adaptive_scm.data import select_split
from adaptive_scm.data.features import feature_columns
from adaptive_scm.forecasting.base import ForecastOutput, Forecaster, rmse
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# PRD-specified quantiles: lower bound, point forecast, upper bound.
_QUANTILES = (0.1, 0.5, 0.9)

# Static categorical columns per PRD Feature 4.
_STATIC_CATEGORICALS = ("item_id", "store_id", "dept_id", "cat_id")

# Checkpoint and metadata filenames inside the save directory.
_CKPT_NAME = "model.ckpt"
_META_NAME = "meta.joblib"


def _sales_derived(column: str) -> bool:
    """Whether an engineered column is derived from the sales target.

    Sales-derived columns (lags, rolling stats) contain future actuals in the
    decoder window, so they must be excluded from TFT inputs (D-4.1). Used by
    :meth:`TFTForecaster._known_real_columns`.

    Args:
        column: Column name from the engineered frame.

    Returns:
        True if the column is a lag or rolling feature of sales.
    """
    return column.startswith("sales_lag_") or column.startswith("sales_roll_")


class TFTForecaster(Forecaster):
    """Temporal Fusion Transformer with P10/P50/P90 quantile output.

    Builds a ``TimeSeriesDataSet`` over the engineered frame with the PRD's
    input partitioning, trains with Adam + quantile loss and early stopping on
    validation quantile loss, then forecasts the post-train horizon in one
    forward pass. ``point_forecast`` is P50, ``lower_bound`` P10, ``upper_bound``
    P90 (sorted to repair any quantile crossing, then floored at zero).
    Interchangeable with ARIMA and XGBoost via the :class:`Forecaster` interface.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        hidden_size: int = 16,
        attention_head_size: int = 4,
        encoder_length: int = 56,
        quantiles: tuple[float, ...] = _QUANTILES,
    ) -> None:
        """Configure model and training hyperparameters.

        Mirrors ``config/forecasters/tft.yaml``; ``encoder_length`` is an
        additive knob (D-4.3) since the PRD does not specify the encoder
        window. No training happens here.

        Args:
            learning_rate: Adam learning rate.
            batch_size: Training batch size.
            max_epochs: Hard cap on training epochs.
            early_stopping_patience: Epochs without val-loss improvement
                before stopping.
            hidden_size: TFT hidden width.
            attention_head_size: Number of attention heads.
            encoder_length: History window (days) the encoder consumes.
            quantiles: Three ascending quantiles (lower, point, upper).

        Raises:
            ValueError: If ``quantiles`` is not three ascending values in (0, 1),
                or any size/patience parameter is non-positive.
        """
        if len(quantiles) != 3 or not all(0 < q < 1 for q in quantiles):
            raise ValueError(f"quantiles must be three values in (0, 1), got {quantiles}")
        if list(quantiles) != sorted(quantiles):
            raise ValueError(f"quantiles must be ascending, got {quantiles}")
        for name, value in (
            ("batch_size", batch_size),
            ("max_epochs", max_epochs),
            ("early_stopping_patience", early_stopping_patience),
            ("hidden_size", hidden_size),
            ("attention_head_size", attention_head_size),
            ("encoder_length", encoder_length),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._max_epochs = int(max_epochs)
        self._patience = int(early_stopping_patience)
        self._hidden_size = int(hidden_size)
        self._attention_heads = int(attention_head_size)
        self._encoder_length = int(encoder_length)
        self._quantiles = tuple(float(q) for q in quantiles)

        self._model = None  # fitted TemporalFusionTransformer
        self._frame: pd.DataFrame | None = None  # slim frame with time_idx
        self._known_reals: list[str] | None = None
        self._train_end: int | None = None  # last train time_idx
        self._prediction_length: int | None = None
        self._historical_rmse: float | None = None
        self._epochs_trained: int | None = None
        self._dataset_params: dict | None = None

    @property
    def historical_rmse(self) -> float:
        """Validation RMSE of the P50 forecast, recorded during :meth:`fit`.

        Returns:
            The validation RMSE as a float.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._historical_rmse is None:
            raise RuntimeError("historical_rmse is unavailable until fit() is called")
        return self._historical_rmse

    @property
    def epochs_trained(self) -> int:
        """Number of completed training epochs (for the convergence check).

        Returns:
            Epochs actually run (early stopping may end below ``max_epochs``).

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        self._require_fitted()
        return int(self._epochs_trained)

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the TFT on the train split with early stopping on the val split.

        Builds a slim frame (time_idx, static IDs, sales, known reals), a
        training ``TimeSeriesDataSet`` whose decoder windows lie entirely inside
        the train split, and a predict-mode validation set over the val horizon.
        Trains with Adam + quantile loss, then records ``historical_rmse`` as the
        P50-vs-actuals RMSE over the validation horizon (consistent with D-2.1).

        Args:
            train_data: Engineered, split-labeled frame from the data pipeline.

        Raises:
            ValueError: If required columns or splits are missing, or the train
                split is shorter than ``encoder_length`` plus the val horizon.
        """
        required = {"sales", "split", "date"}
        if not required.issubset(train_data.columns):
            raise ValueError(f"train_data must contain columns {sorted(required)}")
        train_df = select_split(train_data, "train")
        val_df = select_split(train_data, "val")
        if train_df.empty or val_df.empty:
            raise ValueError("both 'train' and 'val' splits must be non-empty")

        self._prediction_length = len(val_df)
        if len(train_df) < self._encoder_length + self._prediction_length:
            raise ValueError(
                f"train split ({len(train_df)} days) must be >= encoder_length "
                f"({self._encoder_length}) + prediction horizon ({self._prediction_length})"
            )

        self._frame = self._prepare_frame(train_data)
        self._train_end = len(train_df) - 1

        import lightning.pytorch as L
        from lightning.pytorch.callbacks import EarlyStopping
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        training_set = TimeSeriesDataSet(
            self._frame[self._frame["time_idx"] <= self._train_end],
            time_idx="time_idx",
            target="sales",
            group_ids=["series_id"],
            max_encoder_length=self._encoder_length,
            max_prediction_length=self._prediction_length,
            static_categoricals=[c for c in _STATIC_CATEGORICALS if c in self._frame.columns],
            time_varying_known_reals=["time_idx", *self._known_reals],
            time_varying_unknown_reals=["sales"],
        )
        self._dataset_params = training_set.get_parameters()

        # Predict-mode val set: encoder = train tail, decoder = val horizon.
        validation_set = TimeSeriesDataSet.from_dataset(
            training_set,
            self._frame[self._frame["time_idx"] <= self._train_end + self._prediction_length],
            predict=True,
            stop_randomization=True,
        )
        train_dl = training_set.to_dataloader(
            train=True, batch_size=self._batch_size, num_workers=0
        )
        val_dl = validation_set.to_dataloader(
            train=False, batch_size=self._batch_size, num_workers=0
        )

        self._model = TemporalFusionTransformer.from_dataset(
            training_set,
            learning_rate=self._learning_rate,
            hidden_size=self._hidden_size,
            attention_head_size=self._attention_heads,
            loss=QuantileLoss(list(self._quantiles)),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = L.Trainer(
                max_epochs=self._max_epochs,
                callbacks=[EarlyStopping(monitor="val_loss", patience=self._patience)],
                accelerator="auto",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
            )
            trainer.fit(self._model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        self._trainer = trainer
        self._epochs_trained = int(trainer.current_epoch)

        val_point = self.predict(self._prediction_length).point_forecast
        self._historical_rmse = rmse(val_point, val_df["sales"].to_numpy(dtype=float))

        _LOG.info(
            "tft_fitted",
            epochs_trained=self._epochs_trained,
            max_epochs=self._max_epochs,
            historical_rmse=self._historical_rmse,
            n_known_reals=len(self._known_reals),
            encoder_length=self._encoder_length,
        )

    def predict(self, horizon: int) -> ForecastOutput:
        """Forecast ``horizon`` days after training in one forward pass.

        Builds a predict-mode dataset whose decoder covers the first ``horizon``
        post-train days, runs the model in quantile mode, sorts the quantile
        axis to repair any crossing (D-4.2), floors at zero (D-2.2), and maps
        P50/P10/P90 to point/lower/upper.

        Args:
            horizon: Days to forecast. Must be in ``[1, prediction_length]``
                (the model's trained decoder length, normally the 28-day val
                horizon).

        Returns:
            A :class:`ForecastOutput` of length ``horizon``.

        Raises:
            ValueError: If ``horizon`` is out of range.
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if horizon > self._prediction_length:
            raise ValueError(
                f"horizon {horizon} exceeds trained prediction length "
                f"({self._prediction_length}); TFT forecasts in a single pass"
            )
        max_idx = self._train_end + self._prediction_length
        if max_idx >= len(self._frame):
            raise ValueError("stored frame lacks rows for the requested horizon")

        from pytorch_forecasting import TimeSeriesDataSet

        predict_set = TimeSeriesDataSet.from_parameters(
            self._dataset_params,
            self._frame[self._frame["time_idx"] <= max_idx],
            predict=True,
            stop_randomization=True,
        )
        dl = predict_set.to_dataloader(train=False, batch_size=self._batch_size, num_workers=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantile_pred = self._model.predict(
                dl,
                mode="quantiles",
                trainer_kwargs={
                    "logger": False,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
            )

        q = np.asarray(quantile_pred, dtype=float)[0]  # (prediction_length, 3)
        q = np.sort(q, axis=-1)  # repair quantile crossing (D-4.2)
        q = np.clip(q, 0.0, None)[:horizon]

        return ForecastOutput(
            point_forecast=q[:, 1],
            lower_bound=q[:, 0],
            upper_bound=q[:, 2],
            historical_rmse=float(self._historical_rmse) if self._historical_rmse else 0.0,
        )

    def save(self, path: Path) -> None:
        """Persist the model checkpoint and metadata to a directory.

        Writes ``model.ckpt`` (Lightning checkpoint) and ``meta.joblib``
        (stored frame, dataset parameters, hyperparameters, RMSE) under
        ``path``, which is created as a directory. The :class:`Forecaster`
        contract explicitly allows a directory artifact.

        Args:
            path: Destination directory.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._trainer.save_checkpoint(path / _CKPT_NAME)
        joblib.dump(
            {
                "frame": self._frame,
                "known_reals": self._known_reals,
                "train_end": self._train_end,
                "prediction_length": self._prediction_length,
                "historical_rmse": self._historical_rmse,
                "epochs_trained": self._epochs_trained,
                "dataset_parameters": self._dataset_params,
                "batch_size": self._batch_size,
            },
            path / _META_NAME,
        )
        _LOG.info("tft_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "TFTForecaster":
        """Load a forecaster previously written by :meth:`save`.

        Restores the Lightning checkpoint and the metadata sidecar so the
        loaded instance predicts identically without retraining.

        Args:
            path: Directory passed to a prior :meth:`save` call.

        Returns:
            A fitted :class:`TFTForecaster` ready for :meth:`predict`.
        """
        from pytorch_forecasting import TemporalFusionTransformer

        path = Path(path)
        meta = joblib.load(path / _META_NAME)
        instance = cls(batch_size=meta["batch_size"])
        instance._model = TemporalFusionTransformer.load_from_checkpoint(
            path / _CKPT_NAME, map_location="cpu"
        )
        instance._frame = meta["frame"]
        instance._known_reals = meta["known_reals"]
        instance._train_end = meta["train_end"]
        instance._prediction_length = meta["prediction_length"]
        instance._historical_rmse = meta["historical_rmse"]
        instance._epochs_trained = meta["epochs_trained"]
        instance._dataset_params = meta["dataset_parameters"]
        _LOG.info("tft_loaded", path=str(path))
        return instance

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the slim TFT input frame and record known-real columns.

        Selects: a contiguous ``time_idx``, a constant ``series_id`` group key,
        the static ID columns that exist, ``sales`` as float, and every
        non-sales-derived engineered feature as float (calendar, event, price).
        Sales-derived lag/rolling columns are excluded because their decoder
        rows would contain future actuals (D-4.1).

        Args:
            df: Engineered, split-labeled frame.

        Returns:
            Slim frame sorted by date with ``time_idx`` 0..n-1.
        """
        out = df.sort_values("date").reset_index(drop=True).copy()
        out["time_idx"] = np.arange(len(out), dtype=np.int64)
        out["series_id"] = "series_0"

        self._known_reals = [c for c in feature_columns(df) if not _sales_derived(c)]

        keep = ["time_idx", "series_id", "sales", *self._known_reals]
        static_present = [c for c in _STATIC_CATEGORICALS if c in out.columns]
        keep.extend(static_present)

        slim = out[keep].copy()
        slim["sales"] = slim["sales"].astype(float)
        for c in self._known_reals:
            slim[c] = slim[c].astype(float)
        for c in static_present:
            slim[c] = slim[c].astype(str)
        return slim

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted yet.

        Guard shared by :meth:`predict`, :meth:`save`, and the read-only
        properties.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
