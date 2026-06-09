"""Unit tests for the data pipeline (PRD Feature 1).

Covers loader (filter + join + ffill), validator (all three gates), feature
engineer (lags + rolling + calendar + events + price), splitter (positional
1597/28/28), and the top-level ``preprocess`` orchestrator. Uses the
synthetic M5 fixture from ``conftest.py`` — no dependency on the real M5
dataset (per project memory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_scm.data import (
    engineer_features,
    load_m5_series,
    preprocess,
    split_by_position,
    validate_series,
)
from adaptive_scm.data.loader import _snap_column_for_state

# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


class TestLoader:
    def test_loads_only_target_series(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)

        # Exactly one (item, store) pair should appear; the distractor is filtered out.
        assert df["item_id"].nunique() == 1
        assert df["store_id"].nunique() == 1
        assert df.loc[0, "item_id"] == item_id
        assert df.loc[0, "store_id"] == store_id

    def test_long_format_one_row_per_day(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)

        # Number of rows should equal the number of d_* columns in the wide
        # sales file (which is 1700 for the synthetic fixture).
        assert len(df) == 1700
        # Dates should be monotonically increasing.
        assert df["date"].is_monotonic_increasing
        # No duplicate dates.
        assert df["date"].is_unique

    def test_joins_calendar_and_prices(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)

        # Calendar fields present.
        for col in ("wm_yr_wk", "wday", "month", "year", "event_name_1"):
            assert col in df.columns

        # SNAP column was renamed correctly.
        assert "snap" in df.columns
        assert "snap_CA" not in df.columns

        # Prices are joined and non-null after ffill+bfill.
        assert df["sell_price"].notna().all()
        assert (df["sell_price"] > 0).all()

    def test_raises_on_unknown_product(self, synthetic_m5_dir):
        with pytest.raises(ValueError, match="No sales row found"):
            load_m5_series(synthetic_m5_dir, item_id="DOES_NOT_EXIST", store_id="CA_1")

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_m5_series(tmp_path, item_id="X", store_id="Y")

    def test_snap_column_rejects_unknown_state(self):
        with pytest.raises(ValueError, match="Unknown M5 state_id"):
            _snap_column_for_state("XX")


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class TestValidator:
    def test_passes_on_clean_series(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        validate_series(df)  # should not raise

    def test_rejects_high_zero_fraction(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        # Zero out ~20% of the sales column.
        rng = np.random.default_rng(0)
        zero_idx = rng.choice(len(df), size=int(0.2 * len(df)), replace=False)
        df.loc[zero_idx, "sales"] = 0
        with pytest.raises(ValueError, match="zero-sales days"):
            validate_series(df)

    def test_rejects_short_history(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        truncated = df.iloc[:1000].copy()  # <4 years
        with pytest.raises(ValueError, match="history"):
            validate_series(truncated)

    def test_rejects_missing_events(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        df = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        df["event_name_1"] = None
        with pytest.raises(ValueError, match="calendar events"):
            validate_series(df)

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError, match="empty"):
            validate_series(pd.DataFrame({"sales": [], "date": [], "event_name_1": []}))


# --------------------------------------------------------------------------- #
# Feature engineer
# --------------------------------------------------------------------------- #


class TestFeatureEngineer:
    def test_adds_lag_columns(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        for lag in (1, 7, 14, 28, 365):
            assert f"sales_lag_{lag}" in feat.columns
        # First row of the result corresponds to day 366 of the raw series,
        # so the lag_365 value equals the raw sales at row 0.
        assert feat.loc[0, "sales_lag_365"] == raw.loc[0, "sales"]

    def test_adds_rolling_columns_no_leakage(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        for w in (7, 28):
            assert f"sales_roll_mean_{w}" in feat.columns
            assert f"sales_roll_std_{w}" in feat.columns

        # Pick row 100 in the post-drop frame and confirm the 7-day rolling
        # mean equals the mean of the 7 sales values strictly before it in
        # the original frame.
        target_date = feat.loc[100, "date"]
        raw_idx = raw.index[raw["date"] == target_date][0]
        expected = float(raw["sales"].iloc[raw_idx - 7 : raw_idx].mean())
        assert feat.loc[100, "sales_roll_mean_7"] == pytest.approx(expected)

    def test_adds_calendar_features(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        for name in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
            assert f"dow_{name}" in feat.columns
        # Each row has exactly one day-of-week one-hot set.
        dow_cols = [c for c in feat.columns if c.startswith("dow_")]
        assert (feat[dow_cols].sum(axis=1) == 1).all()

        for col in ("day_of_month", "week_of_year", "month_num", "is_month_start", "is_month_end"):
            assert col in feat.columns

    def test_adds_event_features(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        for etype in ("Cultural", "National", "Religious", "Sporting"):
            assert f"event_type_{etype}" in feat.columns
        # At least one event-name one-hot column should exist.
        assert any(c.startswith("event_is_") for c in feat.columns)

    def test_adds_price_features(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        assert "price_index" in feat.columns
        assert "price_change_wk" in feat.columns
        assert "is_promo" in feat.columns
        # price_index has mean ~1 by construction (price / mean price).
        assert feat["price_index"].mean() == pytest.approx(1.0, abs=0.05)
        # is_promo is binary.
        assert set(feat["is_promo"].unique()).issubset({0, 1})

    def test_drops_longest_lag_warmup(self, synthetic_m5_dir, synthetic_item_store):
        item_id, store_id = synthetic_item_store
        raw = load_m5_series(synthetic_m5_dir, item_id=item_id, store_id=store_id)
        feat = engineer_features(raw)
        # 365 rows must be dropped to make the t-365 lag valid everywhere.
        assert len(feat) == len(raw) - 365
        assert feat["sales_lag_365"].notna().all()


# --------------------------------------------------------------------------- #
# Splitter
# --------------------------------------------------------------------------- #


class TestSplitter:
    def test_split_assigns_three_groups_in_order(self):
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=200), "sales": range(200)})
        out = split_by_position(df, train_days=100, val_days=50, test_days=30)
        assert len(out) == 180
        assert (out["split"].iloc[:100] == "train").all()
        assert (out["split"].iloc[100:150] == "val").all()
        assert (out["split"].iloc[150:] == "test").all()

    def test_split_raises_when_too_few_rows(self):
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "sales": range(10)})
        with pytest.raises(ValueError, match="need 100"):
            split_by_position(df, train_days=50, val_days=30, test_days=20)


# --------------------------------------------------------------------------- #
# End-to-end preprocess
# --------------------------------------------------------------------------- #


class TestPreprocess:
    def test_end_to_end_writes_parquet_with_split_column(
        self, synthetic_m5_dir, synthetic_item_store, tmp_path
    ):
        item_id, store_id = synthetic_item_store
        out = preprocess(
            raw_dir=synthetic_m5_dir,
            processed_dir=tmp_path,
            item_id=item_id,
            store_id=store_id,
            train_days=1000,
            val_days=28,
            test_days=28,
        )
        assert out.exists()
        assert out.name == f"{item_id}_{store_id}.parquet"

        df = pd.read_parquet(out)
        # split column present with the three values, no others.
        assert set(df["split"].unique()) == {"train", "val", "test"}
        # Required columns from the acceptance criterion: date, sales, split,
        # plus engineered features.
        for required in ("date", "sales", "split", "sales_lag_1", "price_index"):
            assert required in df.columns
        # Row count matches the split sizes.
        assert len(df) == 1000 + 28 + 28
