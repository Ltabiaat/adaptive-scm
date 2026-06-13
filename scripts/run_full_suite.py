"""CLI entrypoint for the full experimental suite (PRD Feature 11).

Runs all 3x3x3 = 27 ``(forecaster, policy, condition)`` combinations for the
configured replication count, skipping any whose result file already exists
(resumable), then aggregates every experiment's summary into
``results/analysis/full_suite.parquet`` and renders
``results/analysis/summary.md`` with the cost, fill-rate, resilience, and H3
correlation tables.

Usage:
    uv run python scripts/run_full_suite.py --replications=30
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd
from omegaconf import OmegaConf

from adaptive_scm.evaluation import collect_summary_rows, render_summary_markdown
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)
_RESULTS_DIR = Path("results")
_SIM_DIR = _RESULTS_DIR / "simulations"
_ANALYSIS_DIR = _RESULTS_DIR / "analysis"

_FORECASTERS = ("arima", "xgboost", "tft")
_POLICIES = ("eoq", "order_up_to", "ppo")
_CONDITIONS = ("baseline", "demand_spike", "lead_time_disruption")


def _summary_row(path: Path, forecaster: str, policy: str, condition: str) -> dict:
    """Extract the labeled summary row from one experiment's Parquet.

    Reads the single ``record_type == 'summary'`` row and tags it with the
    combination labels so the suite table can be pivoted by condition.

    Args:
        path: Path to an experiment Parquet.
        forecaster: Forecaster label.
        policy: Policy label.
        condition: Condition label.

    Returns:
        A flat dict of the summary metrics plus labels.
    """
    df = pd.read_parquet(path)
    summary = df[df["record_type"] == "summary"].iloc[0].to_dict()
    summary = {k: v for k, v in summary.items() if pd.notna(v) and k != "record_type"}
    summary.update(forecaster=forecaster, policy=policy, condition=condition)
    return summary


@click.command()
@click.option("--replications", default=30, type=int)
@click.option(
    "--config",
    "config_path",
    default="config/default.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--seed", default=42, type=int)
def main(replications: int, config_path: Path, seed: int) -> None:
    """Run (or resume) the full suite and write the aggregate outputs.

    For each of the 27 combinations, runs ``scripts/run_experiment.py`` as a
    subprocess unless its result file already exists (resumable). Then collects
    every summary row, writes ``full_suite.parquet``, and renders ``summary.md``.

    Args:
        replications: Replications per combination.
        config_path: Path to the YAML config.
        seed: Base seed forwarded to each experiment.
    """
    OmegaConf.load(config_path)  # validate config is loadable before the long run
    _SIM_DIR.mkdir(parents=True, exist_ok=True)
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(_FORECASTERS, _POLICIES, _CONDITIONS))
    summaries: list[dict] = []

    for forecaster, policy, condition in combinations:
        out_path = _SIM_DIR / f"{forecaster}_{policy}_{condition}.parquet"
        if out_path.exists():
            _LOG.info("skip_existing", combination=out_path.stem)
        else:
            _LOG.info("running", forecaster=forecaster, policy=policy, condition=condition)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_experiment.py",
                    f"--forecaster={forecaster}",
                    f"--policy={policy}",
                    f"--condition={condition}",
                    f"--replications={replications}",
                    f"--config={config_path}",
                    f"--seed={seed}",
                ],
                check=True,
            )
        summaries.append(_summary_row(out_path, forecaster, policy, condition))

    suite = collect_summary_rows(summaries)
    suite_path = _ANALYSIS_DIR / "full_suite.parquet"
    suite.to_parquet(suite_path, index=False)

    report = render_summary_markdown(suite)
    report_path = _ANALYSIS_DIR / "summary.md"
    report_path.write_text(report)

    _LOG.info(
        "suite_complete",
        combinations=len(combinations),
        suite_table=str(suite_path),
        summary_report=str(report_path),
    )


if __name__ == "__main__":
    main()
