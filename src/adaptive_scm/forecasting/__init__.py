"""Forecasting models. ARIMA, XGBoost, and TFT all implement the ``Forecaster`` ABC.

The base interface and the error helpers are always importable. The concrete
forecasters are loaded **lazily** (PEP 562 ``__getattr__``): a name like
``XGBoostForecaster`` is imported only when first accessed, never at package
import time. This matters on macOS, where XGBoost and PyTorch each bundle their
own OpenMP runtime and loading both into one process segfaults (D-4.7). Lazy
loading lets a single-framework task (e.g. training XGBoost) avoid pulling in
torch at all, so it never triggers the clash.
"""

from adaptive_scm.forecasting.base import (
    ForecastOutput,
    Forecaster,
    forecast_accuracy,
    mape,
    rmse,
)

__all__ = [
    "Forecaster",
    "ForecastOutput",
    "rmse",
    "mape",
    "forecast_accuracy",
    "ARIMAForecaster",
    "XGBoostForecaster",
    "TFTForecaster",
]

_LAZY = {
    "ARIMAForecaster": ("adaptive_scm.forecasting.arima", "ARIMAForecaster"),
    "XGBoostForecaster": ("adaptive_scm.forecasting.xgboost", "XGBoostForecaster"),
    "TFTForecaster": ("adaptive_scm.forecasting.tft", "TFTForecaster"),
}


def __getattr__(name: str):
    """Lazily import a concrete forecaster on first access (PEP 562).

    Looks the name up in the lazy registry and imports only that submodule, so
    importing the package (or one forecaster) never loads the others' heavy
    backends. Raises ``AttributeError`` for unknown names, and a clear
    ``ImportError`` if the backing optional dependency is missing.

    Args:
        name: Attribute being accessed on the package.

    Returns:
        The requested forecaster class.

    Raises:
        AttributeError: If ``name`` is not a known forecaster.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__() -> list[str]:
    """Include the lazy forecasters in ``dir()`` for discoverability."""
    return sorted(__all__)
