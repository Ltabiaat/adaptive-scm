"""Shared pytest fixtures.

Builds tiny but schema-correct synthetic M5 CSVs in a temp directory and
exposes them as fixtures. Tests use these instead of the real ~12M-row M5
dataset so the suite stays fast and CI-friendly (per project memory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Synthetic dimensions. Tuned to be (a) large enough to clear PRD validation
# gates (>= 4 years of history, <10% zero-sales days, >=1 event per year) and
# (b) small enough to keep tests fast.
_N_DAYS = 1700  # ~4.6 years
_START_DATE = pd.Timestamp("2020-01-01")
_ITEM_ID = "FOODS_3_TEST"
_STORE_ID = "CA_1"
_DEPT_ID = "FOODS_3"
_CAT_ID = "FOODS"
_STATE_ID = "CA"


def _build_sales_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Construct a wide-format synthetic sales DataFrame.

    Two rows (the target product-store and a distractor) so loader filtering
    is actually exercised. Daily sales follow a positive Poisson process with
    a weekly seasonal component to keep zero-sales fraction comfortably below
    10%. Used only by fixtures in this module.

    Args:
        rng: NumPy random generator (seeded by the caller).

    Returns:
        Wide-format DataFrame with the M5 id columns plus ``d_1..d_N_DAYS``.
    """
    # Target series: clear weekly seasonality, mean ~6, very few zeros.
    days = np.arange(_N_DAYS)
    weekly = 2.0 * np.sin(2 * np.pi * days / 7)
    lam = np.clip(6.0 + weekly, 0.5, None)
    target_sales = rng.poisson(lam=lam).astype(np.int32)

    # Distractor row so the loader's filter is actually exercised.
    distractor_sales = rng.poisson(lam=3.0, size=_N_DAYS).astype(np.int32)

    d_cols = {f"d_{i + 1}": [target_sales[i], distractor_sales[i]] for i in range(_N_DAYS)}
    base = {
        "id": [f"{_ITEM_ID}_{_STORE_ID}_evaluation", "OTHER_ITEM_CA_2_evaluation"],
        "item_id": [_ITEM_ID, "OTHER_ITEM"],
        "dept_id": [_DEPT_ID, _DEPT_ID],
        "cat_id": [_CAT_ID, _CAT_ID],
        "store_id": [_STORE_ID, "CA_2"],
        "state_id": [_STATE_ID, _STATE_ID],
    }
    return pd.DataFrame({**base, **d_cols})


def _build_calendar_frame() -> pd.DataFrame:
    """Construct a synthetic M5 calendar DataFrame.

    Spans ``_N_DAYS`` days from ``_START_DATE``, includes ``wm_yr_wk`` and
    weekday/month/year fields, sprinkles ~6 calendar events per year (alternating
    types) so the PRD validator's "events per year" check passes, and includes
    all three ``snap_*`` columns. Used only by fixtures in this module.

    Returns:
        DataFrame mirroring the columns of the real M5 ``calendar.csv``.
    """
    dates = pd.date_range(_START_DATE, periods=_N_DAYS, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": _walmart_week_index(dates),
            "d": [f"d_{i + 1}" for i in range(_N_DAYS)],
            "wday": ((dates.dayofweek + 2) % 7) + 1,  # M5 uses Sat=1..Fri=7
            "month": dates.month,
            "year": dates.year,
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )

    # Insert two events per quarter so every calendar year of history gets
    # >= 1 non-null event (PRD validator gate).
    event_types = ["Cultural", "National", "Religious", "Sporting"]
    event_names = ["EventA", "EventB", "EventC", "EventD"]
    quarter_offsets = [10, 100, 200, 300]
    for year_offset in range(0, _N_DAYS, 365):
        for k, offset in enumerate(quarter_offsets):
            idx = year_offset + offset
            if idx >= _N_DAYS:
                break
            df.loc[idx, "event_name_1"] = event_names[k % len(event_names)]
            df.loc[idx, "event_type_1"] = event_types[k % len(event_types)]
    return df


def _walmart_week_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Compute a Walmart-style ``wm_yr_wk`` integer for each date.

    Walmart's fiscal week starts on Saturday. The real M5 calendar encodes
    this as ``year * 100 + week_within_fiscal_year``. This helper produces a
    monotonic integer that satisfies the join key contract for tests, without
    reproducing Walmart's exact fiscal calendar. Used only by
    ``_build_calendar_frame``.

    Args:
        dates: A pandas ``DatetimeIndex``.

    Returns:
        Integer array of week indices, the same length as ``dates``.
    """
    iso = dates.isocalendar()
    return (iso.year.to_numpy() * 100 + iso.week.to_numpy()).astype(np.int32)


def _build_prices_frame(calendar: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Construct a synthetic M5 prices DataFrame.

    One price row per ``(store_id, item_id, wm_yr_wk)`` triple, with random
    fluctuations around a base price plus occasional discount weeks (driving
    the ``is_promo`` feature). Covers both the target and distractor series
    so cross-row filtering is exercised. Used only by the fixture in this
    module.

    Args:
        calendar: The synthetic calendar DataFrame.
        rng: NumPy random generator.

    Returns:
        Long-format DataFrame mirroring real M5 ``sell_prices.csv``.
    """
    unique_weeks = calendar["wm_yr_wk"].drop_duplicates().sort_values().to_numpy()
    rows = []
    for store, item, base in (
        (_STORE_ID, _ITEM_ID, 3.50),
        ("CA_2", "OTHER_ITEM", 2.20),
    ):
        prices = base + rng.normal(0.0, 0.05, size=len(unique_weeks))
        # Promo weeks: 5% chance of a 15% discount.
        promo_mask = rng.random(size=len(unique_weeks)) < 0.05
        prices[promo_mask] *= 0.85
        for wk, price in zip(unique_weeks, prices):
            rows.append((store, item, int(wk), float(round(price, 2))))
    return pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


@pytest.fixture(scope="session")
def synthetic_m5_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a session-scoped synthetic M5 dataset to a temp directory.

    Produces ``sales_train_evaluation.csv``, ``calendar.csv``, and
    ``sell_prices.csv`` files matching the real M5 schema. Other fixtures and
    tests in the data layer point ``raw_dir`` at this directory. Cached at
    session scope so the synthetic data is built once per pytest run.

    Args:
        tmp_path_factory: Pytest's session-scoped temp directory factory.

    Returns:
        Path of the directory holding the three synthetic CSV files.
    """
    rng = np.random.default_rng(42)
    raw_dir = tmp_path_factory.mktemp("m5_raw")

    sales = _build_sales_frame(rng)
    calendar = _build_calendar_frame()
    prices = _build_prices_frame(calendar, rng)

    sales.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw_dir / "calendar.csv", index=False)
    prices.to_csv(raw_dir / "sell_prices.csv", index=False)
    return raw_dir


@pytest.fixture(scope="session")
def synthetic_item_store() -> tuple[str, str]:
    """Return the ``(item_id, store_id)`` pair present in the synthetic dataset.

    Single source of truth for the target series identifiers so tests do not
    hard-code constants that drift from the fixture.

    Returns:
        ``(item_id, store_id)`` tuple.
    """
    return _ITEM_ID, _STORE_ID
