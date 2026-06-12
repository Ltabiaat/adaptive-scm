"""Unit tests for forecasters.

Phase 2 covers the ARIMA forecaster (PRD Feature 2). The expensive
``auto_arima`` fit is done once via a session-scoped fixture on a trimmed
synthetic series with tight search bounds, so the suite stays fast. Tests
verify interface compliance, forecast shape/contract, the zero floor,
validation RMSE, and save/load round-trip identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_scm.data import engineer_features, load_m5_series, select_split, split_by_position
from adaptive_scm.forecasting import (
    ARIMAForecaster,
    ForecastOutput,
    Forecaster,
    XGBoostForecaster,
)

_HORIZON = 28


@pytest.fixture(scope="session")
def arima_training_frame(synthetic_m5_dir, synthetic_item_store) -> pd.DataFrame:
    """A small train/val/test frame for fast ARIMA fitting.

    Loads the synthetic series and trims it to ~300 days so the order search
    is quick, then applies a positional split. ARIMA consumes only ``sales``
    and ``split``, so feature engineering is skipped here for speed. Session
    scoped because several tests share the same fitted model.

    Returns:
        DataFrame with ``date``, ``sales``, and ``split`` columns.
    """
    item_id, store_id = synthetic_item_store
    raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
    small = raw.iloc[:300][["date", "sales"]].copy()
    return split_by_position(small, train_days=250, val_days=28, test_days=22)


@pytest.fixture(scope="session")
def fitted_arima(arima_training_frame) -> ARIMAForecaster:
    """A session-scoped ARIMA fitted once with tight search bounds.

    Bounds (``max_p=max_q=2``, ``max_P=max_Q=max_d=max_D=1``) keep the
    stepwise search to a few seconds while still exercising the seasonal path
    (``m=7``). Shared across tests to avoid repeated fits.

    Returns:
        A fitted :class:`ARIMAForecaster`.
    """
    forecaster = ARIMAForecaster(
        seasonal=True,
        seasonal_period=7,
        information_criterion="aic",
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_d=1,
        max_D=1,
        stepwise=True,
    )
    forecaster.fit(arima_training_frame)
    return forecaster


def test_arima(fitted_arima):
    """Top-level acceptance test named in PRD Feature 2 acceptance criteria.

    Asserts the fitted model implements the interface and emits a contract-
    valid 28-day forecast with a populated validation RMSE.
    """
    assert isinstance(fitted_arima, Forecaster)

    out = fitted_arima.predict(_HORIZON)
    assert isinstance(out, ForecastOutput)
    assert out.horizon == _HORIZON
    assert out.point_forecast.shape == (_HORIZON,)
    assert out.lower_bound is not None and out.lower_bound.shape == (_HORIZON,)
    assert out.upper_bound is not None and out.upper_bound.shape == (_HORIZON,)
    assert out.historical_rmse > 0


class TestARIMAInterface:
    def test_selects_seasonal_order_with_m7(self, fitted_arima):
        # Seasonal period must be 7 in the selected seasonal order tuple.
        assert fitted_arima.seasonal_order[-1] == 7
        # Non-seasonal order is a 3-tuple, seasonal order a 4-tuple.
        assert len(fitted_arima.order) == 3
        assert len(fitted_arima.seasonal_order) == 4

    def test_rejects_invalid_seasonal_period(self):
        with pytest.raises(ValueError, match="seasonal_period"):
            ARIMAForecaster(seasonal_period=0)

    def test_predict_before_fit_raises(self):
        forecaster = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="not fitted"):
            forecaster.predict(_HORIZON)

    def test_historical_rmse_before_fit_raises(self):
        forecaster = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="historical_rmse"):
            _ = forecaster.historical_rmse

    def test_rejects_non_positive_horizon(self, fitted_arima):
        with pytest.raises(ValueError, match="horizon"):
            fitted_arima.predict(0)

    def test_fit_requires_sales_column(self):
        forecaster = ARIMAForecaster()
        with pytest.raises(ValueError, match="sales"):
            forecaster.fit(pd.DataFrame({"not_sales": [1, 2, 3]}))


class TestARIMAForecastContract:
    def test_forecast_is_non_negative(self, fitted_arima):
        # Floor-at-zero invariant for point and both bounds.
        out = fitted_arima.predict(_HORIZON)
        assert (out.point_forecast >= 0).all()
        assert (out.lower_bound >= 0).all()
        assert (out.upper_bound >= 0).all()

    def test_bounds_bracket_point_forecast(self, fitted_arima):
        # Where the lower bound has not been clipped to zero, the interval
        # must bracket the point forecast.
        out = fitted_arima.predict(_HORIZON)
        assert (out.upper_bound >= out.point_forecast - 1e-6).all()
        unclipped = out.lower_bound > 0
        assert (out.lower_bound[unclipped] <= out.point_forecast[unclipped] + 1e-6).all()

    def test_validation_rmse_is_reasonable(self, fitted_arima):
        # Sanity check (not a strict target, per the acceptance criterion):
        # RMSE should be a finite positive value on the order of the series'
        # own scale (mean ~6), not wildly large.
        rmse = fitted_arima.historical_rmse
        assert np.isfinite(rmse)
        assert 0 < rmse < 50

    def test_horizon_independent_of_default(self, fitted_arima):
        # The model can forecast horizons other than 28.
        out_short = fitted_arima.predict(7)
        assert out_short.horizon == 7


class TestARIMASaveLoad:
    def test_round_trip_preserves_predictions(self, fitted_arima, tmp_path):
        # Acceptance criterion: save/load round-trip without loss of state.
        path = tmp_path / "arima.joblib"
        fitted_arima.save(path)
        assert path.exists()

        loaded = ARIMAForecaster.load(path)
        assert isinstance(loaded, ARIMAForecaster)

        before = fitted_arima.predict(_HORIZON)
        after = loaded.predict(_HORIZON)
        np.testing.assert_allclose(before.point_forecast, after.point_forecast)
        np.testing.assert_allclose(before.lower_bound, after.lower_bound)
        np.testing.assert_allclose(before.upper_bound, after.upper_bound)
        assert before.historical_rmse == pytest.approx(after.historical_rmse)

    def test_round_trip_preserves_orders(self, fitted_arima, tmp_path):
        path = tmp_path / "arima2.joblib"
        fitted_arima.save(path)
        loaded = ARIMAForecaster.load(path)
        assert loaded.order == fitted_arima.order
        assert loaded.seasonal_order == fitted_arima.seasonal_order

    def test_save_before_fit_raises(self, tmp_path):
        forecaster = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="not fitted"):
            forecaster.save(tmp_path / "x.joblib")


class TestARIMAFallbackRMSE:
    def test_in_sample_rmse_without_val_split(self, arima_training_frame):
        # When no 'split' column is present, fit on all rows and fall back to
        # in-sample residual RMSE so historical_rmse is still populated.
        no_split = arima_training_frame[["date", "sales"]].copy()
        forecaster = ARIMAForecaster(max_p=1, max_q=1, max_P=1, max_Q=1, max_d=1, max_D=1)
        forecaster.fit(no_split)
        assert np.isfinite(forecaster.historical_rmse)
        assert forecaster.historical_rmse > 0


# --------------------------------------------------------------------------- #
# XGBoost (Feature 3)
# --------------------------------------------------------------------------- #

# Small grid so the 81-point default doesn't run in unit tests.
_TEST_GRID = {
    "max_depth": (3, 6),
    "learning_rate": (0.05, 0.1),
    "n_estimators": (200,),
    "reg_lambda": (0.1, 1.0),
}


@pytest.fixture(scope="session")
def xgb_training_frame(synthetic_m5_dir, synthetic_item_store) -> pd.DataFrame:
    """Engineered, split-labeled frame for XGBoost (train_days=600 seeds lag_365).

    Runs the full feature pipeline (XGBoost needs the engineered columns) and
    uses 600 train days so the recursive forecaster's lag-365 buffer is
    seedable. Session scoped to share across XGBoost tests.

    Returns:
        DataFrame with engineered features and a ``split`` column.
    """
    item_id, store_id = synthetic_item_store
    raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
    feat = engineer_features(raw)
    return split_by_position(feat, train_days=600, val_days=28, test_days=28)


@pytest.fixture(scope="session")
def fitted_xgb(xgb_training_frame) -> XGBoostForecaster:
    """A session-scoped XGBoost fitted once with the small test grid.

    Returns:
        A fitted :class:`XGBoostForecaster`.
    """
    forecaster = XGBoostForecaster(grid=_TEST_GRID, early_stopping_rounds=20)
    forecaster.fit(xgb_training_frame)
    return forecaster


def test_xgboost(fitted_xgb):
    """Top-level acceptance test named in PRD Feature 3 acceptance criteria.

    Asserts interface compliance, a contract-valid 28-day forecast, and that
    the best grid hyperparameters are recorded.
    """
    assert isinstance(fitted_xgb, Forecaster)

    out = fitted_xgb.predict(_HORIZON)
    assert isinstance(out, ForecastOutput)
    assert out.horizon == _HORIZON
    assert out.point_forecast.shape == (_HORIZON,)
    assert out.historical_rmse > 0

    # Best hyperparameters logged/persisted (acceptance criterion).
    params = fitted_xgb.best_params
    assert set(params) >= {"max_depth", "learning_rate", "n_estimators", "reg_lambda"}


class TestXGBoostInterface:
    def test_rejects_non_positive_patience(self):
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            XGBoostForecaster(early_stopping_rounds=0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            XGBoostForecaster().predict(_HORIZON)

    def test_historical_rmse_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="historical_rmse"):
            _ = XGBoostForecaster().historical_rmse

    def test_fit_requires_sales_and_split(self):
        with pytest.raises(ValueError, match="sales"):
            XGBoostForecaster(grid=_TEST_GRID).fit(pd.DataFrame({"x": [1, 2, 3]}))

    def test_rejects_non_positive_horizon(self, fitted_xgb):
        with pytest.raises(ValueError, match="horizon"):
            fitted_xgb.predict(0)

    def test_rejects_horizon_beyond_future_rows(self, fitted_xgb):
        with pytest.raises(ValueError, match="exceeds"):
            fitted_xgb.predict(10_000)


class TestXGBoostForecastContract:
    def test_uses_full_feature_set(self, fitted_xgb):
        # Engineered features present, raw passthrough excluded (D-3.1).
        cols = fitted_xgb._feature_columns
        assert "sales_lag_7" in cols
        assert "price_index" in cols
        assert any(c.startswith("dow_") for c in cols)
        for excluded in ("sales", "split", "sell_price", "month", "year", "snap"):
            assert excluded not in cols

    def test_forecast_is_non_negative(self, fitted_xgb):
        out = fitted_xgb.predict(_HORIZON)
        assert (out.point_forecast >= 0).all()
        assert (out.lower_bound >= 0).all()
        assert (out.upper_bound >= 0).all()

    def test_ci_is_symmetric_rmse_band(self, fitted_xgb):
        # Bounds are point ± 1.96 * historical_rmse, floored at zero (D-3.5).
        out = fitted_xgb.predict(_HORIZON)
        margin = 1.96 * fitted_xgb.historical_rmse
        expected_upper = out.point_forecast + margin
        np.testing.assert_allclose(out.upper_bound, expected_upper, rtol=1e-6)

    def test_recursive_forecast_is_deterministic(self, fitted_xgb):
        # Same fitted model -> identical recursive forecast across calls.
        a = fitted_xgb.predict(_HORIZON)
        b = fitted_xgb.predict(_HORIZON)
        np.testing.assert_array_equal(a.point_forecast, b.point_forecast)

    def test_recursion_feeds_predictions_forward(self, fitted_xgb):
        # Sanity: a longer horizon shares its prefix with a shorter one, since
        # the recursion is path-deterministic from the same buffer seed.
        short = fitted_xgb.predict(7).point_forecast
        long = fitted_xgb.predict(14).point_forecast
        np.testing.assert_allclose(short, long[:7], rtol=1e-6)


class TestXGBoostBeatsNaive:
    def test_outperforms_seasonal_naive(self, fitted_xgb, xgb_training_frame):
        # Acceptance criterion: XGBoost one-step val RMSE < naive sales[t-7].
        val = select_split(xgb_training_frame, "val")
        y_val = val["sales"].to_numpy(dtype=float)
        x_val = val[fitted_xgb._feature_columns].to_numpy(dtype=float)

        xgb_rmse = float(np.sqrt(np.mean((fitted_xgb._model.predict(x_val) - y_val) ** 2)))
        naive_rmse = float(np.sqrt(np.mean((val["sales_lag_7"].to_numpy(float) - y_val) ** 2)))
        assert xgb_rmse < naive_rmse


class TestXGBoostSaveLoad:
    def test_round_trip_preserves_predictions(self, fitted_xgb, tmp_path):
        path = tmp_path / "xgb.joblib"
        fitted_xgb.save(path)
        assert path.exists()

        loaded = XGBoostForecaster.load(path)
        before = fitted_xgb.predict(_HORIZON)
        after = loaded.predict(_HORIZON)
        np.testing.assert_allclose(before.point_forecast, after.point_forecast)
        assert before.historical_rmse == pytest.approx(after.historical_rmse)
        assert loaded.best_params == fitted_xgb.best_params

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not fitted"):
            XGBoostForecaster().save(tmp_path / "x.joblib")


# --------------------------------------------------------------------------- #
# TFT (Feature 4)
# --------------------------------------------------------------------------- #

# Tiny network + few epochs: the test verifies the contract, not accuracy.
_TFT_TEST_KWARGS = dict(
    hidden_size=4,
    attention_head_size=1,
    max_epochs=3,
    early_stopping_patience=10,
    batch_size=32,
    encoder_length=28,
)


@pytest.fixture(scope="session")
def tft_training_frame(synthetic_m5_dir, synthetic_item_store) -> pd.DataFrame:
    """Engineered, split-labeled frame sized for a fast TFT fit.

    300 train days keeps epochs quick while satisfying
    ``encoder_length + horizon`` and giving the encoder several seasonal
    cycles. Session scoped to share across TFT tests.

    Returns:
        DataFrame with engineered features and a ``split`` column.
    """
    item_id, store_id = synthetic_item_store
    raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
    feat = engineer_features(raw)
    return split_by_position(feat, train_days=300, val_days=28, test_days=28)


@pytest.fixture(scope="session")
def fitted_tft(tft_training_frame):
    """A session-scoped TFT fitted once with the tiny test configuration.

    Returns:
        A fitted :class:`TFTForecaster`.
    """
    from adaptive_scm.forecasting import TFTForecaster
    from adaptive_scm.utils.seeding import set_global_seed

    set_global_seed(42)
    forecaster = TFTForecaster(**_TFT_TEST_KWARGS)
    forecaster.fit(tft_training_frame)
    return forecaster


def test_tft(fitted_tft):
    """Top-level acceptance test named in PRD Feature 4 acceptance criteria.

    Asserts interface compliance, a P10/P50/P90 forecast of the expected
    shape, and convergence within the epoch cap.
    """
    from adaptive_scm.forecasting import TFTForecaster

    assert isinstance(fitted_tft, Forecaster)
    assert isinstance(fitted_tft, TFTForecaster)

    out = fitted_tft.predict(_HORIZON)
    assert isinstance(out, ForecastOutput)
    assert out.horizon == _HORIZON
    # ForecastOutput carries P10/P50/P90 as lower/point/upper (PRD criterion).
    assert out.lower_bound is not None and out.lower_bound.shape == (_HORIZON,)
    assert out.upper_bound is not None and out.upper_bound.shape == (_HORIZON,)
    assert out.historical_rmse > 0

    # "Training converges within max epochs" criterion: training completed
    # and did not exceed the cap.
    assert 1 <= fitted_tft.epochs_trained <= _TFT_TEST_KWARGS["max_epochs"]


class TestTFTConstruction:
    def test_rejects_bad_quantiles(self):
        from adaptive_scm.forecasting import TFTForecaster

        with pytest.raises(ValueError, match="quantiles"):
            TFTForecaster(quantiles=(0.1, 0.5))
        with pytest.raises(ValueError, match="ascending"):
            TFTForecaster(quantiles=(0.9, 0.5, 0.1))

    def test_rejects_non_positive_sizes(self):
        from adaptive_scm.forecasting import TFTForecaster

        with pytest.raises(ValueError, match="hidden_size"):
            TFTForecaster(hidden_size=0)
        with pytest.raises(ValueError, match="encoder_length"):
            TFTForecaster(encoder_length=0)


class TestTFTInterface:
    def test_predict_before_fit_raises(self):
        from adaptive_scm.forecasting import TFTForecaster

        with pytest.raises(RuntimeError, match="not fitted"):
            TFTForecaster().predict(_HORIZON)

    def test_historical_rmse_before_fit_raises(self):
        from adaptive_scm.forecasting import TFTForecaster

        with pytest.raises(RuntimeError, match="historical_rmse"):
            _ = TFTForecaster().historical_rmse

    def test_rejects_non_positive_horizon(self, fitted_tft):
        with pytest.raises(ValueError, match="horizon"):
            fitted_tft.predict(0)

    def test_rejects_horizon_beyond_prediction_length(self, fitted_tft):
        # TFT forecasts in a single pass capped at the trained decoder length.
        with pytest.raises(ValueError, match="single pass"):
            fitted_tft.predict(_HORIZON + 1)

    def test_fit_requires_train_and_val(self, tft_training_frame):
        from adaptive_scm.forecasting import TFTForecaster

        only_train = tft_training_frame[tft_training_frame["split"] == "train"]
        with pytest.raises(ValueError, match="non-empty"):
            TFTForecaster(**_TFT_TEST_KWARGS).fit(only_train)


class TestTFTForecastContract:
    def test_quantiles_are_ordered(self, fitted_tft):
        # P10 <= P50 <= P90 after the crossing repair (D-4.2).
        out = fitted_tft.predict(_HORIZON)
        assert (out.lower_bound <= out.point_forecast + 1e-6).all()
        assert (out.point_forecast <= out.upper_bound + 1e-6).all()

    def test_forecast_is_non_negative(self, fitted_tft):
        out = fitted_tft.predict(_HORIZON)
        assert (out.point_forecast >= 0).all()
        assert (out.lower_bound >= 0).all()

    def test_excludes_sales_derived_features(self, fitted_tft):
        # Lag/rolling columns would leak future actuals in the decoder (D-4.1).
        assert not any(
            c.startswith("sales_lag_") or c.startswith("sales_roll_")
            for c in fitted_tft._known_reals
        )
        # But known covariates (calendar/event/price) are present.
        assert "price_index" in fitted_tft._known_reals
        assert any(c.startswith("dow_") for c in fitted_tft._known_reals)

    def test_shorter_horizon_is_prefix(self, fitted_tft):
        # Single-pass forecast: a shorter horizon is a slice of the full one.
        short = fitted_tft.predict(7).point_forecast
        full = fitted_tft.predict(_HORIZON).point_forecast
        np.testing.assert_allclose(short, full[:7], rtol=1e-6)

    def test_prediction_is_deterministic(self, fitted_tft):
        a = fitted_tft.predict(_HORIZON).point_forecast
        b = fitted_tft.predict(_HORIZON).point_forecast
        np.testing.assert_array_equal(a, b)


class TestTFTSaveLoad:
    def test_round_trip_preserves_predictions(self, fitted_tft, tmp_path):
        from adaptive_scm.forecasting import TFTForecaster

        path = tmp_path / "tft_model"
        fitted_tft.save(path)
        assert (path / "model.ckpt").exists()
        assert (path / "meta.joblib").exists()

        loaded = TFTForecaster.load(path)
        before = fitted_tft.predict(_HORIZON)
        after = loaded.predict(_HORIZON)
        np.testing.assert_allclose(before.point_forecast, after.point_forecast, rtol=1e-5)
        np.testing.assert_allclose(before.lower_bound, after.lower_bound, rtol=1e-5)
        np.testing.assert_allclose(before.upper_bound, after.upper_bound, rtol=1e-5)
        assert before.historical_rmse == pytest.approx(after.historical_rmse)

    def test_save_before_fit_raises(self, tmp_path):
        from adaptive_scm.forecasting import TFTForecaster

        with pytest.raises(RuntimeError, match="not fitted"):
            TFTForecaster().save(tmp_path / "x")
