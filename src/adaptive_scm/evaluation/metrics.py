"""Trajectory metrics for the inventory simulation.

Turns a replication's per-day records (the ``info`` dicts the environment emits,
augmented with reward) into the cost and service metrics the experiment runner
reports, plus the two resilience metrics (service-level degradation and recovery
time) used under disruption conditions. ``aggregate_metrics`` reduces a list of
per-replication metric dicts to means and standard deviations.
"""

from __future__ import annotations

import numpy as np


def compute_episode_metrics(records: list[dict]) -> dict[str, float]:
    """Compute cost and service metrics for one replication.

    Sums the per-day costs into totals and derives fill rate and stockout
    frequency from demand and lost sales. Resilience metrics (degradation,
    recovery) are intentionally *not* computed here: under the Chapter-3
    definition they are cross-condition (baseline vs disruption) and so are
    computed at the suite level in ``evaluation.resilience`` (D-10.3).

    Args:
        records: Per-day dicts with keys ``demand``, ``lost_sales``,
            ``holding_cost``, ``stockout_cost``, ``order_cost`` (the env ``info``
            plus reward). One entry per simulated day.

    Returns:
        Dict with ``total_cost``, ``holding_cost``, ``stockout_cost``,
        ``order_cost``, ``fill_rate``, and ``stockout_frequency``.
    """
    demand = np.array([r["demand"] for r in records], dtype=float)
    lost = np.array([r["lost_sales"] for r in records], dtype=float)
    holding = np.array([r["holding_cost"] for r in records], dtype=float)
    stockout = np.array([r["stockout_cost"] for r in records], dtype=float)
    order = np.array([r["order_cost"] for r in records], dtype=float)

    sales = demand - lost
    total_demand = float(demand.sum())
    fill_rate = float(sales.sum() / total_demand) if total_demand > 0 else 1.0
    stockout_frequency = float((lost > 1e-9).mean())

    return {
        "total_cost": float(holding.sum() + stockout.sum() + order.sum()),
        "holding_cost": float(holding.sum()),
        "stockout_cost": float(stockout.sum()),
        "order_cost": float(order.sum()),
        "fill_rate": fill_rate,
        "stockout_frequency": stockout_frequency,
    }


def aggregate_metrics(per_replication: list[dict]) -> dict[str, float]:
    """Reduce per-replication metrics to means and standard deviations.

    Computes the mean of every metric across replications, plus the standard
    deviation of ``total_cost`` (the headline metric, reported with dispersion
    per PRD Feature 10). Called once per experiment after all replications run.

    Args:
        per_replication: List of metric dicts from :func:`compute_episode_metrics`.

    Returns:
        Dict with ``{metric}_mean`` for every metric and ``total_cost_std``.

    Raises:
        ValueError: If ``per_replication`` is empty.
    """
    if not per_replication:
        raise ValueError("no replications to aggregate")
    keys = per_replication[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in per_replication], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
    out["total_cost_std"] = float(
        np.array([m["total_cost"] for m in per_replication], dtype=float).std(ddof=0)
    )
    out["n_replications"] = float(len(per_replication))
    return out
