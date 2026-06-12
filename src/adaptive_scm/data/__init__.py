"""M5 data loading, preprocessing, and feature engineering.

Public functions:
    - ``load_m5_series``: read raw M5 files and return a long-format DataFrame
      for one ``(item_id, store_id)`` pair.
    - ``engineer_features``: add lag, rolling, calendar, event, and price features.
    - ``preprocess``: end-to-end orchestration producing the final processed Parquet.
"""

from adaptive_scm.data.features import engineer_features, feature_columns
from adaptive_scm.data.loader import load_m5_series, validate_series
from adaptive_scm.data.preprocessor import preprocess, select_split, split_by_position

__all__ = [
    "load_m5_series",
    "validate_series",
    "engineer_features",
    "feature_columns",
    "preprocess",
    "select_split",
    "split_by_position",
]
