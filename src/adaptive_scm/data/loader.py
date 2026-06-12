"""M5 raw-data loader.

Reads the three M5 CSV files (sales, calendar, prices), filters to a single
``(item_id, store_id)`` pair, joins them into a long-format daily DataFrame,
and validates the resulting series against the PRD's quality gates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# Column names defined by the M5 dataset schema.
SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

_SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5_series(
    raw_dir: Path | str,
    item_id: str,
    store_id: str,
) -> pd.DataFrame:
    """Load and join M5 data for a single product-store pair.

    Reads ``sales_train_evaluation.csv``, ``calendar.csv``, and
    ``sell_prices.csv`` from ``raw_dir``; filters sales to the row matching
    ``(item_id, store_id)``; melts the wide ``d_*`` columns into long format;
    joins calendar (by ``d``) and prices (by ``store_id``, ``item_id``,
    ``wm_yr_wk``); and forward-fills missing prices. The result is the
    foundation for the feature engineering step.

    Args:
        raw_dir: Directory containing the three raw M5 CSV files.
        item_id: M5 item identifier (e.g. ``"FOODS_3_090"``).
        store_id: M5 store identifier (e.g. ``"CA_1"``).

    Returns:
        Long-format DataFrame with one row per day, sorted by ``date`` ascending.
        Columns: ``date`` (datetime64), ``d`` (str), ``item_id``, ``store_id``,
        ``dept_id``, ``cat_id``, ``state_id``, ``sales`` (int), ``wm_yr_wk``,
        ``wday``, ``month``, ``year``, ``event_name_1``, ``event_type_1``,
        ``event_name_2``, ``event_type_2``, ``snap`` (int, the state-specific
        SNAP flag), ``sell_price`` (float, forward-filled).

    Raises:
        FileNotFoundError: If any of the three required files is missing.
        ValueError: If no row matches ``(item_id, store_id)`` in the sales file.
    """
    raw_path = Path(raw_dir)
    sales = _read_required(raw_path / SALES_FILE)
    calendar = _read_required(raw_path / CALENDAR_FILE)
    prices = _read_required(raw_path / PRICES_FILE)

    mask = (sales["item_id"] == item_id) & (sales["store_id"] == store_id)
    series_rows = sales.loc[mask]
    if series_rows.empty:
        raise ValueError(
            f"No sales row found for item_id={item_id!r}, store_id={store_id!r} in {SALES_FILE}"
        )
    if len(series_rows) > 1:
        # Defensive: M5 has exactly one row per (item, store), but guard anyway.
        raise ValueError(
            f"Expected exactly one sales row for item={item_id}, store={store_id}, "
            f"got {len(series_rows)}"
        )

    d_cols = [c for c in sales.columns if c.startswith("d_")]
    long_sales = series_rows.melt(
        id_vars=_SALES_ID_COLS,
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    long_sales["sales"] = long_sales["sales"].astype(np.int32)

    calendar_keep = [
        "date",
        "d",
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]
    snap_col = _snap_column_for_state(series_rows.iloc[0]["state_id"])
    calendar_keep.append(snap_col)
    merged = long_sales.merge(calendar[calendar_keep], on="d", how="left")
    merged = merged.rename(columns={snap_col: "snap"})

    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill prices (per PRD), then back-fill leading NaNs that precede
    # the first observed price record for this series.
    merged["sell_price"] = merged["sell_price"].ffill().bfill()

    _LOG.info(
        "loaded_m5_series",
        item_id=item_id,
        store_id=store_id,
        rows=len(merged),
        date_min=str(merged["date"].min().date()),
        date_max=str(merged["date"].max().date()),
    )
    return merged


def validate_series(df: pd.DataFrame) -> None:
    """Validate that a loaded series meets the PRD's quality gates.

    Checks three conditions: less than 10% of days have zero sales, at least
    four full years of history, and at least one non-null calendar event per
    year of history. Called by ``preprocess`` before feature engineering so
    selection of a degenerate product fails fast with a clear message.

    Args:
        df: Long-format DataFrame returned by :func:`load_m5_series`.

    Raises:
        ValueError: If any of the three quality gates is violated.
    """
    n = len(df)
    if n == 0:
        raise ValueError("series is empty")

    zero_frac = float((df["sales"] == 0).mean())
    if zero_frac >= 0.10:
        raise ValueError(f"series has {zero_frac:.1%} zero-sales days (>=10% threshold)")

    span_days = (df["date"].max() - df["date"].min()).days + 1
    if span_days < 4 * 365:
        raise ValueError(f"series has only {span_days} days of history (<4 years required)")

    # Interpretation: "≥ 1 promotional event per year" → at least one row with
    # a non-null M5 calendar event per calendar year of history. M5 calendar
    # events are dataset-wide so this gate is loose, but it catches truncated
    # series where the event columns were stripped.
    events_per_year = (
        df.assign(_year=df["date"].dt.year)
        .groupby("_year")["event_name_1"]
        .apply(lambda s: s.notna().sum())
    )
    bad_years = events_per_year[events_per_year < 1]
    if not bad_years.empty:
        raise ValueError(f"series has years with zero calendar events: {bad_years.index.tolist()}")

    _LOG.info(
        "series_validated",
        rows=n,
        zero_sales_fraction=zero_frac,
        span_days=span_days,
    )


def _read_required(path: Path) -> pd.DataFrame:
    """Read a CSV file, raising a clear error if it is missing.

    Thin wrapper around ``pd.read_csv`` used to surface missing-file failures
    with the absolute path that was attempted. Called only by ``load_m5_series``.

    Args:
        path: Absolute path to a CSV file.

    Returns:
        DataFrame parsed from ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required M5 file not found: {path}")
    return pd.read_csv(path)


def _snap_column_for_state(state_id: str) -> str:
    """Return the SNAP indicator column name for a given M5 state.

    M5's calendar has three SNAP columns (``snap_CA``, ``snap_TX``,
    ``snap_WI``); the one relevant to a series depends on which state the
    store is in. Used by :func:`load_m5_series` to select the right SNAP
    column before merging.

    Args:
        state_id: M5 state identifier (``"CA"``, ``"TX"``, or ``"WI"``).

    Returns:
        Column name like ``"snap_CA"``.

    Raises:
        ValueError: If ``state_id`` is not one of the three M5 states.
    """
    state_id = str(state_id).upper()
    if state_id not in {"CA", "TX", "WI"}:
        raise ValueError(f"Unknown M5 state_id: {state_id!r}")
    return f"snap_{state_id}"
