"""Trajectory metrics for the inventory simulation.

Turns a replication's per-day records (the ``info`` dicts the environment emits,
augmented with reward) into the cost and service metrics the experiment runner
reports, plus the two resilience metrics (service-level degradation and recovery
time) used under disruption conditions. ``aggregate_metrics`` reduces a list of
per-replication metric dicts to means and standard deviations.
"""

from __future__ import annotations

import numpy as np

# Daily-service tolerance and smoothing window for recovery-time detection.
_RECOVERY_TOL = 0.02
_RECOVERY_SMOOTH = 3


def compute_episode_metrics(
    records: list[dict],
    disruption_window: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Compute cost, service, and resilience metrics for one replication.

    Sums the per-day costs into totals, derives fill rate and stockout frequency
    from demand and lost sales, and (when a disruption window is given) computes
    service-level degradation and recovery time from the daily service series.
    Called once per replication by the runner.

    Args:
        records: Per-day dicts with keys ``demand``, ``lost_sales``,
            ``holding_cost``, ``stockout_cost``, ``order_cost`` (the env ``info``
            plus reward). One entry per simulated day.
        disruption_window: Optional ``(start_day, end_day)`` half-open interval
            of the disruption. ``None`` for the baseline condition (degradation
            and recovery are then 0).

    Returns:
        Dict with ``total_cost``, ``holding_cost``, ``stockout_cost``,
        ``order_cost``, ``fill_rate``, ``stockout_frequency``,
        ``service_level_degradation``, and ``recovery_time``.
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

    degradation, recovery = _resilience(demand, lost, disruption_window)

    return {
        "total_cost": float(holding.sum() + stockout.sum() + order.sum()),
        "holding_cost": float(holding.sum()),
        "stockout_cost": float(stockout.sum()),
        "order_cost": float(order.sum()),
        "fill_rate": fill_rate,
        "stockout_frequency": stockout_frequency,
        "service_level_degradation": degradation,
        "recovery_time": recovery,
    }


def _daily_service(demand: np.ndarray, lost: np.ndarray) -> np.ndarray:
    """Per-day service level (fraction of demand met).

    Returns ``(demand - lost) / demand`` per day, with days of zero demand
    treated as fully served (service 1.0). Used by the resilience metrics.

    Args:
        demand: Per-day demand.
        lost: Per-day lost sales.

    Returns:
        Per-day service-level array in ``[0, 1]``.
    """
    served = demand - lost
    return np.where(demand > 1e-9, served / np.maximum(demand, 1e-9), 1.0)


def _resilience(
    demand: np.ndarray,
    lost: np.ndarray,
    window: tuple[int, int] | None,
) -> tuple[float, float]:
    """Compute service-level degradation and recovery time.

    Degradation is the drop in mean daily service during the disruption window
    relative to the pre-window period. Recovery time is the number of days after
    the window ends until the (3-day smoothed) daily service returns to within a
    tolerance of the pre-window level; if it never recovers, it is the number of
    remaining days. Both are 0 when no window is given (baseline).

    Args:
        demand: Per-day demand.
        lost: Per-day lost sales.
        window: Optional ``(start, end)`` half-open disruption interval.

    Returns:
        Tuple ``(service_level_degradation, recovery_time)``.
    """
    if window is None:
        return 0.0, 0.0
    start, end = window
    n = len(demand)
    start = max(0, min(start, n))
    end = max(start, min(end, n))

    service = _daily_service(demand, lost)
    pre = service[:start]
    inside = service[start:end]
    pre_level = float(pre.mean()) if len(pre) > 0 else float(service.mean())
    in_level = float(inside.mean()) if len(inside) > 0 else pre_level
    degradation = max(0.0, pre_level - in_level)

    post = service[end:]
    if len(post) == 0:
        return degradation, 0.0
    smoothed = _smooth(post, _RECOVERY_SMOOTH)
    recovered = np.where(smoothed >= pre_level - _RECOVERY_TOL)[0]
    recovery_time = float(recovered[0]) if len(recovered) > 0 else float(len(post))
    return degradation, recovery_time


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average used to denoise the daily service series.

    Args:
        values: Series to smooth.
        window: Trailing window length.

    Returns:
        Smoothed array, same length as ``values``.
    """
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


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
