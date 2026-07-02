"""Report held-out test-set forecast accuracy (RMSE and MAPE) per forecaster.

Loads each trained forecaster, generates its forecast over the held-out test
window, compares to the realized test-split sales, and writes both accuracy
metrics from Section 3.7 to ``results/analysis/forecast_accuracy.md``. RMSE
penalizes large errors (most relevant to inventory cost); MAPE is the
scale-independent complement and excludes zero-demand days. This closes the one
metric the suite does not produce: the suite's ``forecast_rmse`` is the
validation RMSE used for noise calibration, whereas this reports test-set RMSE
and MAPE on the final 28 days for direct reporting.

Usage:
    uv run python scripts/report_forecast_accuracy.py
"""

from __future__ import annotations

from adaptive_scm.utils.openmp import allow_duplicate_openmp

# A forecaster load may pull in XGBoost or torch; permit OpenMP coexistence on
# macOS before either backend imports (D-4.8). Harmless elsewhere.
allow_duplicate_openmp()

from pathlib import Path  # noqa: E402

import click  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from adaptive_scm.forecasting import Forecaster, forecast_accuracy  # noqa: E402
from adaptive_scm.utils.logging import get_logger  # noqa: E402

_LOG = get_logger(__name__)
_RESULTS_DIR = Path("results")
_ANALYSIS_DIR = _RESULTS_DIR / "analysis"
_FORECASTERS = ("arima", "xgboost", "tft")


def _load_forecaster(name: str) -> Forecaster:
    """Load a frozen forecaster, importing only its backend (D-4.7).

    Imports the single requested forecaster's submodule so a non-torch
    forecaster never loads torch (and vice versa), matching the loaders in the
    experiment scripts.

    Args:
        name: One of ``arima``, ``xgboost``, ``tft``.

    Returns:
        The loaded forecaster.

    Raises:
        FileNotFoundError: If the artifact is missing.
    """
    paths = {
        "arima": _RESULTS_DIR / "forecaster_arima.joblib",
        "xgboost": _RESULTS_DIR / "forecaster_xgboost.joblib",
        "tft": _RESULTS_DIR / "forecaster_tft",
    }
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(f"forecaster {name!r} not found at {path}; train it first")
    cls: type[Forecaster]
    if name == "arima":
        from adaptive_scm.forecasting.arima import ARIMAForecaster

        cls = ARIMAForecaster
    elif name == "xgboost":
        from adaptive_scm.forecasting.xgboost import XGBoostForecaster

        cls = XGBoostForecaster
    else:
        from adaptive_scm.forecasting.tft import TFTForecaster

        cls = TFTForecaster
    return cls.load(path)


@click.command()
@click.option("--config", "config_path", default="config/default.yaml", help="Config file.")
def main(config_path: str) -> None:
    """Compute test-set RMSE and MAPE for each forecaster and write a report.

    For each forecaster: loads it, forecasts the test horizon, and scores the
    prediction against the realized test-split sales using ``forecast_accuracy``.
    Writes a markdown table to ``results/analysis/forecast_accuracy.md``.

    Args:
        config_path: Path to the merged config (for the product-store pair and
            test horizon).
    """
    cfg = OmegaConf.load(config_path)
    item_id = cfg.data.product_store.item_id
    store_id = cfg.data.product_store.store_id
    horizon = int(cfg.simulation.episode.length)

    processed = Path(cfg.data.processed_dir) / f"{item_id}_{store_id}.parquet"
    if not processed.exists():
        raise click.ClickException(f"processed data not found at {processed}; run preprocess first")
    df = pd.read_parquet(processed)
    actual = df[df["split"] == "test"].head(horizon)["sales"].to_numpy(dtype=float)

    rows = []
    for name in _FORECASTERS:
        try:
            forecaster = _load_forecaster(name)
        except FileNotFoundError:
            _LOG.warning("forecaster_missing", forecaster=name)
            click.echo(f"skipping {name}: not trained yet")
            continue
        predicted = np.asarray(forecaster.predict(horizon).point_forecast, dtype=float)
        metrics = forecast_accuracy(predicted, actual)
        rows.append({"forecaster": name, "rmse": metrics["rmse"], "mape": metrics["mape"]})
        _LOG.info("accuracy_scored", forecaster=name, rmse=metrics["rmse"], mape=metrics["mape"])

    if not rows:
        raise click.ClickException("no trained forecasters found; train them first")

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ANALYSIS_DIR / "forecast_accuracy.md"
    lines = [
        "# Forecast Accuracy (held-out test set)",
        "",
        f"Product-store: {item_id} / {store_id}. Test horizon: {horizon} days.",
        "RMSE penalizes large errors; MAPE excludes zero-demand days (Section 3.7).",
        "",
        table.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    out_path.write_text("\n".join(lines))
    _LOG.info("forecast_accuracy_report", path=str(out_path), forecasters=len(rows))
    click.echo(table.to_string(index=False))
    click.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
