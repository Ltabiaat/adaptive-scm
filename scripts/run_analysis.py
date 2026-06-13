"""CLI entrypoint for hypothesis testing (PRD Feature 12).

Loads the per-cell simulation results plus the suite summary, runs the three
hypothesis tests (H1: PPO vs classical; H2: integrated TFT+PPO vs standalone
baselines; H3: forecast accuracy vs cost), and writes
``results/analysis/hypothesis_tests.md`` with p-values, effect sizes, and
interpretations.

Usage:
    uv run python scripts/run_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from adaptive_scm.evaluation import (
    render_hypothesis_markdown,
    test_h1,
    test_h2,
    test_h3,
)
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)
_RESULTS_DIR = Path("results")
_SIM_DIR = _RESULTS_DIR / "simulations"
_ANALYSIS_DIR = _RESULTS_DIR / "analysis"


@click.command()
@click.option(
    "--sim-dir",
    default=_SIM_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of per-cell simulation Parquets.",
)
@click.option(
    "--suite",
    "suite_path",
    default=_ANALYSIS_DIR / "full_suite.parquet",
    type=click.Path(path_type=Path),
    help="Path to the aggregated suite summary Parquet (for H3).",
)
@click.option("--h2-condition", default="baseline", help="Condition to test H2 at.")
def main(sim_dir: Path, suite_path: Path, h2_condition: str) -> None:
    """Run the three hypothesis tests and write the Markdown report.

    Args:
        sim_dir: Directory of per-cell Parquets (for H1/H2 paired tests).
        suite_path: Suite summary Parquet (for the H3 correlation).
        h2_condition: Condition at which to evaluate H2.

    Raises:
        FileNotFoundError: If the simulation directory does not exist.
    """
    if not sim_dir.exists():
        raise FileNotFoundError(f"simulation directory not found: {sim_dir}")

    h1 = test_h1(sim_dir)
    h2 = test_h2(sim_dir, condition=h2_condition)

    if suite_path.exists():
        suite = pd.read_parquet(suite_path)
    else:
        _LOG.warning("suite_missing", path=str(suite_path), note="H3 will be skipped")
        suite = pd.DataFrame()
    h3 = test_h3(suite)

    report = render_hypothesis_markdown(h1, h2, h3)
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ANALYSIS_DIR / "hypothesis_tests.md"
    out_path.write_text(report)

    _LOG.info(
        "analysis_complete",
        h1_comparisons=len(h1),
        h2_comparisons=len(h2),
        h3_rho=h3["spearman_rho"],
        report=str(out_path),
    )


if __name__ == "__main__":
    main()
