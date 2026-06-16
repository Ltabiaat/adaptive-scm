"""Forecasting models. ARIMA, XGBoost, and TFT all implement the ``Forecaster`` ABC.

The base interface is always importable. Concrete forecasters depend on the
optional ``forecasting`` / ``deep`` dependency groups, so they are imported
defensively: if the backing library is absent, the symbol is simply not
exported rather than breaking ``import adaptive_scm.forecasting``.
"""

from adaptive_scm.forecasting.base import (
    ForecastOutput,
    Forecaster,
    forecast_accuracy,
    mape,
    rmse,
)

__all__ = ["Forecaster", "ForecastOutput", "rmse", "mape", "forecast_accuracy"]

try:
    from adaptive_scm.forecasting.arima import ARIMAForecaster  # noqa: F401

    __all__.append("ARIMAForecaster")
except ImportError:  # pragma: no cover - exercised only without the forecasting extra
    pass

try:
    from adaptive_scm.forecasting.xgboost import XGBoostForecaster  # noqa: F401

    __all__.append("XGBoostForecaster")
except ImportError:  # pragma: no cover - exercised only without the forecasting extra
    pass

try:
    from adaptive_scm.forecasting.tft import TFTForecaster  # noqa: F401

    __all__.append("TFTForecaster")
except ImportError:  # pragma: no cover - exercised only without the deep extra
    pass
