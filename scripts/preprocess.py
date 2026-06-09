"""CLI entrypoint for M5 preprocessing.

Reads ``config/default.yaml`` (or a path passed via ``--config``), runs the
full :func:`adaptive_scm.data.preprocess` pipeline for the configured
product-store pair, and writes the result to ``data/processed/``.

Usage:
    uv run python scripts/preprocess.py
    uv run python scripts/preprocess.py --config path/to/config.yaml
"""

from __future__ import annotations

from pathlib import Path

import click
from omegaconf import OmegaConf

from adaptive_scm.data import preprocess
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config/default.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML config.",
)
def main(config_path: Path) -> None:
    """Run preprocessing for the product-store pair in the config.

    Loads the config with OmegaConf, then delegates to
    :func:`adaptive_scm.data.preprocess`. Prints the output Parquet path on
    success. Returns nothing.

    Args:
        config_path: Path to a YAML file matching the schema in PRD Section 5.
    """
    cfg = OmegaConf.load(config_path)
    out_path = preprocess(
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        item_id=cfg.data.product_store.item_id,
        store_id=cfg.data.product_store.store_id,
        train_days=cfg.data.splits.train_days,
        val_days=cfg.data.splits.val_days,
        test_days=cfg.data.splits.test_days,
    )
    _LOG.info("preprocess_done", output=str(out_path))


if __name__ == "__main__":
    main()
