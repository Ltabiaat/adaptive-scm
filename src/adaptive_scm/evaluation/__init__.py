"""Evaluation: trajectory metrics and cross-run analysis.

Public surface:
    - ``compute_episode_metrics``: metrics for one replication's trajectory.
    - ``aggregate_metrics``: mean/std across replications.
    - ``EpisodeMetrics``: the per-episode metric keys (documentation helper).
"""

from adaptive_scm.evaluation.analyzer import (
    collect_summary_rows,
    render_summary_markdown,
    rmse_cost_correlation,
)
from adaptive_scm.evaluation.metrics import (
    aggregate_metrics,
    compute_episode_metrics,
)

__all__ = [
    "compute_episode_metrics",
    "aggregate_metrics",
    "collect_summary_rows",
    "rmse_cost_correlation",
    "render_summary_markdown",
]
