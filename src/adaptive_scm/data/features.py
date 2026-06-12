"""Feature engineering for M5 series.

Pure functions that consume the long-format DataFrame from
:func:`adaptive_scm.data.loader.load_m5_series` and add the lag, rolling,
calendar, event, and price features specified in PRD Feature 1. ARIMA ignores
these features (it consumes only the ``sales`` column); XGBoost and TFT
consume the full set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# PRD-specified lag and rolling-window sizes. Public so the recursive
# XGBoost forecaster can reuse them when rebuilding sales-derived features.
LAGS = (1, 7, 14, 28, 365)
ROLLING_WINDOWS = (7, 28)
_LAGS = LAGS
_ROLLING_WINDOWS = ROLLING_WINDOWS

# M5 has four event types (Cultural, National, Religious, Sporting); these
# become four binary flags. Hard-coded because the M5 schema is fixed.
_EVENT_TYPES = ("Cultural", "National", "Religious", "Sporting")

# Threshold for the promotional flag, per PRD Feature 1.
_PROMO_THRESHOLD = 0.95

# Raw / id / target / passthrough columns that are never model inputs. The
# model feature set is every numeric column NOT in this set (see D-3.1).
NON_FEATURE_COLUMNS = frozenset(
    {
        "date",
        "d",
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "sales",
        "split",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "sell_price",
        "snap",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    }
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model-ready feature columns of an engineered frame.

    Selects every numeric column that is not in :data:`NON_FEATURE_COLUMNS`,
    preserving DataFrame column order. Shared by the XGBoost forecaster (and
    later TFT) so the train matrix and the recursive-forecast rows always use
    the same columns in the same order.

    Args:
        df: A DataFrame produced by :func:`engineer_features`.

    Returns:
        Ordered list of feature column names.
    """
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, calendar, event, and price features.

    Builds the full feature matrix consumed by the XGBoost and TFT forecasters.
    Lag and rolling features use only past sales (no leakage); rolling stats
    are shifted by one day so the value at row ``t`` summarizes days strictly
    before ``t``. Rows with NaN engineered features (the first 365 days for the
    longest lag) are dropped at the end.

    Args:
        df: Long-format DataFrame from :func:`load_m5_series`. Must contain
            ``date``, ``sales``, ``sell_price``, ``wday``, ``month``,
            ``event_name_1``, and ``event_type_1`` columns.

    Returns:
        New DataFrame, sorted by ``date``, with the original columns plus all
        engineered features. The number of rows is reduced by 365 (the longest
        lag) relative to the input.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    _add_lag_features(out)
    _add_rolling_features(out)
    _add_calendar_features(out)
    _add_event_features(out)
    _add_price_features(out)

    pre_rows = len(out)
    out = out.dropna(subset=[f"sales_lag_{max(_LAGS)}"]).reset_index(drop=True)
    _LOG.info(
        "features_engineered",
        rows_in=pre_rows,
        rows_out=len(out),
        rows_dropped=pre_rows - len(out),
        n_columns=out.shape[1],
    )
    return out


def _add_lag_features(df: pd.DataFrame) -> None:
    """Add ``sales_lag_{k}`` columns for each ``k`` in ``_LAGS``.

    Standard time-series lag features: ``sales_lag_k[t] = sales[t-k]``. Used by
    XGBoost and TFT to capture autoregressive structure and weekly/annual
    seasonality. Mutates ``df`` in place because feature steps are chained
    inside :func:`engineer_features`.

    Args:
        df: DataFrame with a ``sales`` column.
    """
    for k in _LAGS:
        df[f"sales_lag_{k}"] = df["sales"].shift(k)


def _add_rolling_features(df: pd.DataFrame) -> None:
    """Add rolling-window mean and std of sales.

    For each window size in ``_ROLLING_WINDOWS``, computes the rolling mean
    and standard deviation, then shifts by one day so the value at ``t``
    depends only on ``[t-window, t-1]`` — no leakage. Used by XGBoost and TFT
    as smoothed level/volatility signals.

    Args:
        df: DataFrame with a ``sales`` column.
    """
    for w in _ROLLING_WINDOWS:
        rolled = df["sales"].rolling(window=w, min_periods=w)
        df[f"sales_roll_mean_{w}"] = rolled.mean().shift(1)
        df[f"sales_roll_std_{w}"] = rolled.std().shift(1)


def _add_calendar_features(df: pd.DataFrame) -> None:
    """Add calendar-derived features: day-of-week one-hot, day-of-month, etc.

    Adds: seven ``dow_*`` one-hot flags (Mon..Sun), ``day_of_month``,
    ``week_of_year``, ``month_num``, ``is_month_start``, ``is_month_end``.
    These capture deterministic calendar effects that the ML models would
    otherwise have to learn through interactions of raw date components.

    Args:
        df: DataFrame with a ``date`` column of dtype ``datetime64``.
    """
    dates = pd.to_datetime(df["date"])
    dow = dates.dt.dayofweek  # Monday=0..Sunday=6
    for i, name in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun")):
        df[f"dow_{name}"] = (dow == i).astype(np.int8)
    df["day_of_month"] = dates.dt.day.astype(np.int16)
    df["week_of_year"] = dates.dt.isocalendar().week.astype(np.int16)
    df["month_num"] = dates.dt.month.astype(np.int16)
    df["is_month_start"] = dates.dt.is_month_start.astype(np.int8)
    df["is_month_end"] = dates.dt.is_month_end.astype(np.int8)


def _add_event_features(df: pd.DataFrame) -> None:
    """Add event-type binary flags and one-hot encoding of specific events.

    Adds four binary flags ``event_type_{Cultural,National,Religious,Sporting}``
    from ``event_type_1``, plus one-hot columns ``event_is_{name}`` for each
    unique non-null value in ``event_name_1``. The secondary ``event_name_2``
    column is ignored (it is sparse and overlaps with the primary slot).

    Args:
        df: DataFrame with ``event_type_1`` and ``event_name_1`` columns.
    """
    for etype in _EVENT_TYPES:
        df[f"event_type_{etype}"] = (df["event_type_1"] == etype).astype(np.int8)

    # One-hot specific event names. Uses only event_name_1 (event_name_2 is
    # extremely sparse and overlaps with the primary slot in M5).
    name_dummies = pd.get_dummies(df["event_name_1"], prefix="event_is", dtype=np.int8)
    for col in name_dummies.columns:
        df[col] = name_dummies[col].to_numpy()


def _add_price_features(df: pd.DataFrame) -> None:
    """Add relative price index, weekly price change, and promotional flag.

    Computes a per-series mean price and adds: ``price_index`` (price divided
    by the mean), ``price_change_wk`` (week-over-week pct change), and
    ``is_promo`` (1 when ``price_index < 0.95``). Captures the dominant price
    effects M5 forecasts respond to.

    Args:
        df: DataFrame with a ``sell_price`` column.
    """
    mean_price = float(df["sell_price"].mean())
    if mean_price <= 0 or not np.isfinite(mean_price):
        raise ValueError(f"non-positive or non-finite mean price: {mean_price}")
    df["price_index"] = (df["sell_price"] / mean_price).astype(np.float32)

    # Weekly price change: 7-day lag is the canonical reference for M5
    # because prices are weekly. ``fillna(0)`` for the first week avoids NaN
    # leakage; downstream rows are dropped anyway by the longest-lag filter.
    price_lag7 = df["sell_price"].shift(7)
    df["price_change_wk"] = ((df["sell_price"] - price_lag7) / price_lag7).fillna(0.0)

    df["is_promo"] = (df["price_index"] < _PROMO_THRESHOLD).astype(np.int8)
