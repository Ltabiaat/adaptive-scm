"""End-to-end preprocessing pipeline.

Composes the loader, validator, feature engineer, and time-ordered splitter
into a single ``preprocess`` function. Output is a Parquet file at
``data/processed/{item_id}_{store_id}.parquet`` carrying a ``split`` column
with values ``"train"``, ``"val"``, or ``"test"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_scm.data.features import engineer_features
from adaptive_scm.data.loader import load_m5_series, validate_series
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the rows of ``df`` belonging to a named split.

    Thin convenience over ``df.loc[df["split"] == split]`` so forecasters share
    one spelling of split selection. Used by the ARIMA and XGBoost forecasters.

    Args:
        df: DataFrame carrying a ``split`` column.
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        The matching rows (a view-backed copy is not made; index is preserved).

    Raises:
        KeyError: If ``df`` has no ``split`` column.
    """
    if "split" not in df.columns:
        raise KeyError("DataFrame has no 'split' column")
    return df.loc[df["split"] == split]


def split_by_position(
    df: pd.DataFrame,
    train_days: int,
    val_days: int,
    test_days: int,
) -> pd.DataFrame:
    """Attach a time-ordered ``split`` column to ``df``.

    Assigns the last ``test_days`` rows to ``"test"``, the ``val_days`` rows
    immediately before them to ``"val"``, and the preceding ``train_days`` rows
    to ``"train"``. Any earlier rows are dropped. Used by :func:`preprocess` to
    produce the canonical 1597/28/28 split from the PRD.

    Args:
        df: DataFrame sorted ascending by date.
        train_days: Number of training rows (most recent before val).
        val_days: Number of validation rows (between train and test).
        test_days: Number of test rows (the tail).

    Returns:
        DataFrame whose last ``train_days + val_days + test_days`` rows carry
        a ``split`` column.

    Raises:
        ValueError: If ``df`` has fewer than ``train_days + val_days + test_days`` rows.
    """
    needed = train_days + val_days + test_days
    if len(df) < needed:
        raise ValueError(
            f"need {needed} rows for split (train={train_days}, val={val_days}, "
            f"test={test_days}); df has {len(df)}"
        )
    out = df.iloc[-needed:].copy().reset_index(drop=True)
    split = np.empty(needed, dtype=object)
    split[:train_days] = "train"
    split[train_days : train_days + val_days] = "val"
    split[train_days + val_days :] = "test"
    out["split"] = split
    return out


def preprocess(
    raw_dir: Path | str,
    processed_dir: Path | str,
    item_id: str,
    store_id: str,
    train_days: int = 1597,
    val_days: int = 28,
    test_days: int = 28,
) -> Path:
    """Run the full preprocessing pipeline and persist the result.

    Loads the raw M5 files, validates the series, engineers features, applies
    the time-ordered split, and writes the result to
    ``{processed_dir}/{item_id}_{store_id}.parquet``. This is the single
    entrypoint called by ``scripts/preprocess.py`` and by the CLI.

    Args:
        raw_dir: Directory containing the three raw M5 CSV files.
        processed_dir: Directory where the processed Parquet will be written.
        item_id: M5 item identifier.
        store_id: M5 store identifier.
        train_days: Number of training rows in the split.
        val_days: Number of validation rows.
        test_days: Number of test rows.

    Returns:
        Path of the written Parquet file.
    """
    raw_df = load_m5_series(raw_dir, item_id=item_id, store_id=store_id)
    validate_series(raw_df)
    feat_df = engineer_features(raw_df)
    split_df = split_by_position(
        feat_df,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
    )

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    out_file = processed_path / f"{item_id}_{store_id}.parquet"
    split_df.to_parquet(out_file, index=False)
    _LOG.info(
        "preprocessed_series",
        path=str(out_file),
        rows=len(split_df),
        columns=split_df.shape[1],
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
    )
    return out_file
