"""Cross-run analysis for the full experimental suite.

Aggregates the per-experiment summary rows into one table and renders a Markdown
report with cost, fill-rate, and resilience tables plus the Spearman correlation
between forecast RMSE and inventory cost (the H3 question). Consumed by
``scripts/run_full_suite.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


def collect_summary_rows(per_experiment: list[dict]) -> pd.DataFrame:
    """Build the suite-level summary table from per-experiment summaries.

    Each input dict is one experiment's aggregate metrics plus its
    ``forecaster``/``policy``/``condition`` labels and ``forecast_rmse``. The
    result has one row per ``(forecaster, policy, condition)`` combination.

    Args:
        per_experiment: List of per-experiment summary dicts.

    Returns:
        A DataFrame, one row per combination, sorted by the label columns.
    """
    df = pd.DataFrame(per_experiment)
    sort_cols = [c for c in ("forecaster", "policy", "condition") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def rmse_cost_correlation(summary: pd.DataFrame) -> tuple[float, float]:
    """Spearman correlation between forecast RMSE and total cost (H3).

    Uses one ``(forecast_rmse, total_cost_mean)`` point per experiment across
    all combinations. A weak or non-significant correlation is itself the H3
    finding: forecast accuracy need not translate into decision quality.

    Args:
        summary: The suite summary table.

    Returns:
        Tuple ``(spearman_rho, p_value)``; ``(nan, nan)`` if undefined.
    """
    if "forecast_rmse" not in summary or len(summary) < 3:
        return float("nan"), float("nan")
    rmse = summary["forecast_rmse"]
    cost = summary["total_cost_mean"]
    # Spearman is undefined when either input is constant (e.g. a single
    # forecaster), so guard against it rather than emit a runtime warning.
    if rmse.nunique() < 2 or cost.nunique() < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(rmse, cost)
    return float(rho), float(p)


def _pivot(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot one metric into a (forecaster, policy) x condition table.

    Args:
        summary: The suite summary table.
        value: The metric column to lay out.

    Returns:
        A pivot table; empty if the column is absent.
    """
    if value not in summary:
        return pd.DataFrame()
    return summary.pivot_table(index=["forecaster", "policy"], columns="condition", values=value)


def render_summary_markdown(summary: pd.DataFrame) -> str:
    """Render the suite summary as a Markdown report.

    Produces total-cost, fill-rate, and resilience tables (each pivoted by
    condition) and a short H3 correlation section. Used to write
    ``results/analysis/summary.md``.

    Args:
        summary: The suite summary table.

    Returns:
        The report as a Markdown string.
    """
    lines = ["# Experimental Suite Summary", ""]
    lines.append(f"Combinations: {len(summary)}")
    lines.append("")

    sections = [
        ("Total cost (mean)", "total_cost_mean"),
        ("Fill rate (mean)", "fill_rate_mean"),
        ("Service-level degradation (mean)", "service_level_degradation_mean"),
        ("Recovery time (mean, days)", "recovery_time_mean"),
    ]
    for title, col in sections:
        table = _pivot(summary, col)
        if table.empty:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(table.round(3).to_markdown())
        lines.append("")

    rho, p = rmse_cost_correlation(summary)
    lines.append("## H3 — forecast accuracy vs inventory cost")
    lines.append("")
    if np.isnan(rho):
        lines.append("Not enough data to compute the RMSE-cost correlation.")
    else:
        lines.append(
            f"Spearman correlation between forecast RMSE and mean total cost: "
            f"rho = {rho:.3f} (p = {p:.3f}). A weak or non-significant value "
            f"supports H3 — forecast accuracy does not directly determine "
            f"decision quality."
        )
    lines.append("")
    return "\n".join(lines)
