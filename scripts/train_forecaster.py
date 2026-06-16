"""CLI entrypoint for training a forecaster.

Loads the project config, reads the preprocessed Parquet for the configured
product-store pair, fits the requested forecaster on the train split, and saves
it under ``results/``. Phase 2 implements ``--model=arima``; ``xgboost`` and
``tft`` raise a clear not-yet-implemented error until their phases land.

Usage:
    uv run python scripts/train_forecaster.py --model=arima
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from omegaconf import OmegaConf

from adaptive_scm.forecasting import Forecaster
from adaptive_scm.utils.logging import get_logger
from adaptive_scm.utils.seeding import set_global_seed

_LOG = get_logger(__name__)

_RESULTS_DIR = Path("results")


def _build_arima(cfg) -> Forecaster:
    """Construct an :class:`ARIMAForecaster` from the merged config.

    Reads the ``forecasters.arima`` block (with ``.get`` fallbacks for the
    optional search-bound knobs) and returns an unfitted forecaster. Isolated
    into a helper so adding XGBoost/TFT builders later keeps ``main`` readable.

    Args:
        cfg: OmegaConf config node for the whole project.

    Returns:
        An unfitted :class:`ARIMAForecaster`.
    """
    a = cfg.forecasters.arima
    from adaptive_scm.forecasting.arima import ARIMAForecaster

    return ARIMAForecaster(
        seasonal=a.seasonal,
        seasonal_period=a.seasonal_period,
        information_criterion=a.information_criterion,
        max_p=a.get("max_p", 5),
        max_q=a.get("max_q", 5),
        max_P=a.get("max_P", 2),
        max_Q=a.get("max_Q", 2),
        max_d=a.get("max_d", 2),
        max_D=a.get("max_D", 1),
        stepwise=a.get("stepwise", True),
    )


def _build_xgboost(cfg) -> Forecaster:
    """Construct an :class:`XGBoostForecaster` from the merged config.

    Reads the ``forecasters.xgboost`` block, translating the YAML grid lists
    into the tuple-valued grid the forecaster expects. Isolated as a helper to
    keep ``main`` readable as more forecasters are added.

    Args:
        cfg: OmegaConf config node for the whole project.

    Returns:
        An unfitted :class:`XGBoostForecaster`.
    """
    x = cfg.forecasters.xgboost
    from adaptive_scm.forecasting.xgboost import XGBoostForecaster

    grid = {
        "max_depth": tuple(x.grid_search.max_depth),
        "learning_rate": tuple(x.grid_search.learning_rate),
        "n_estimators": tuple(x.grid_search.n_estimators),
        "reg_lambda": tuple(float(v) for v in x.grid_search.reg_lambda),
    }
    return XGBoostForecaster(
        grid=grid,
        early_stopping_rounds=x.early_stopping_rounds,
    )


def _build_tft(cfg) -> Forecaster:
    """Construct a :class:`TFTForecaster` from the merged config.

    Reads the ``forecasters.tft`` block. ``encoder_length`` falls back to 56
    days if absent (D-4.3). Isolated as a helper alongside the ARIMA and
    XGBoost builders.

    Args:
        cfg: OmegaConf config node for the whole project.

    Returns:
        An unfitted :class:`TFTForecaster`.
    """
    t = cfg.forecasters.tft
    from adaptive_scm.forecasting.tft import TFTForecaster

    return TFTForecaster(
        learning_rate=t.learning_rate,
        batch_size=t.batch_size,
        max_epochs=t.max_epochs,
        early_stopping_patience=t.early_stopping_patience,
        hidden_size=t.hidden_size,
        attention_head_size=t.attention_head_size,
        encoder_length=t.get("encoder_length", 56),
        accelerator=t.get("accelerator", "cpu"),
        quantiles=tuple(t.quantiles),
    )


@click.command()
@click.option(
    "--model",
    type=click.Choice(["arima", "xgboost", "tft"]),
    required=True,
    help="Which forecaster to train.",
)
@click.option(
    "--config",
    "config_path",
    default="config/default.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML config.",
)
@click.option("--seed", default=42, type=int, help="Global random seed.")
def main(model: str, config_path: Path, seed: int) -> None:
    """Train the requested forecaster and save it under ``results/``.

    Seeds all RNGs, loads the processed Parquet for the configured
    product-store pair, fits the forecaster, logs its validation RMSE, and
    persists it to ``results/forecaster_{model}.joblib``.

    Args:
        model: Forecaster key (``arima`` implemented in Phase 2).
        config_path: Path to the YAML config.
        seed: Global random seed for reproducibility.

    Raises:
        NotImplementedError: If ``model`` is not yet implemented.
        FileNotFoundError: If the processed Parquet does not exist.
    """
    set_global_seed(seed)
    cfg = OmegaConf.load(config_path)

    item_id = cfg.data.product_store.item_id
    store_id = cfg.data.product_store.store_id
    processed = Path(cfg.data.processed_dir) / f"{item_id}_{store_id}.parquet"
    if not processed.exists():
        raise FileNotFoundError(
            f"processed data not found at {processed}; run scripts/preprocess.py first"
        )
    df = pd.read_parquet(processed)

    if model == "arima":
        forecaster = _build_arima(cfg)
    elif model == "xgboost":
        forecaster = _build_xgboost(cfg)
    elif model == "tft":
        forecaster = _build_tft(cfg)
    else:
        raise NotImplementedError(f"forecaster {model!r} is not implemented yet")

    forecaster.fit(df)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # TFT saves a directory (checkpoint + metadata); the others a single file.
    suffix = "" if model == "tft" else ".joblib"
    out_path = _RESULTS_DIR / f"forecaster_{model}{suffix}"
    forecaster.save(out_path)
    _LOG.info(
        "trained_forecaster",
        model=model,
        output=str(out_path),
        historical_rmse=forecaster.historical_rmse,
    )


if __name__ == "__main__":
    main()
