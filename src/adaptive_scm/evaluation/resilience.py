"""Cross-condition resilience metrics (matches Chapter 3, Section 3.7).

Service-level degradation and recovery time are defined *between* the baseline
and a disruption condition, so they cannot be computed from a single run. These
functions take a disruption cell's daily trajectory plus the baseline and
disruption fill rates and produce:

* **service-level degradation** -- the absolute drop in fill rate from the
  baseline condition to the disruption condition.
* **recovery time** -- the number of days after the disruption window ends until
  the (smoothed, replication-averaged) daily service returns to within one
  percentage point of the baseline fill rate.

This replaces the earlier within-run definition (superseded D-10.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Recovery is reached within this fill-rate tolerance of baseline (1 pp, Sec 3.7).
RECOVERY_TOLERANCE = 0.01
# Trailing smoothing applied to the replication-averaged daily-service curve.
RECOVERY_SMOOTHING = 3


def mean_daily_service(daily: pd.DataFrame) -> np.ndarray:
    """Replication-averaged daily service level from a cell's daily rows.

    For each episode day, averages ``(demand - lost_sales) / demand`` across
    replications (days of zero demand count as fully served). Averaging over the
    replications denoises the curve so recovery detection is stable.

    Args:
        daily: Daily rows for one experiment cell, with ``day``, ``demand``,
            and ``lost_sales`` columns (the ``record_type == 'daily'`` rows).

    Returns:
        Array of mean daily service, indexed by episode day.
    """
    rows = daily[daily.get("record_type", "daily") == "daily"].copy()
    served = rows["demand"] - rows["lost_sales"]
    rows["service"] = np.where(rows["demand"] > 1e-9, served / rows["demand"].clip(lower=1e-9), 1.0)
    return rows.groupby("day")["service"].mean().to_numpy()


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average used to denoise the daily-service curve.

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


def compute_resilience(
    disruption_daily: pd.DataFrame,
    baseline_fill_rate: float,
    disruption_fill_rate: float,
    window: tuple[int, int],
    tolerance: float = RECOVERY_TOLERANCE,
) -> dict[str, float]:
    """Compute cross-condition degradation and recovery time for one disruption cell.

    Degradation is ``baseline_fill_rate - disruption_fill_rate`` (clamped at 0).
    Recovery time is the number of days after the disruption window ends until the
    smoothed replication-averaged daily service first reaches
    ``baseline_fill_rate - tolerance``; if it never does, it is the number of days
    remaining after the window. Both reference the baseline condition, per
    Section 3.7.

    Args:
        disruption_daily: Daily rows of the disruption cell.
        baseline_fill_rate: Mean fill rate of the matching baseline cell.
        disruption_fill_rate: Mean fill rate of this disruption cell.
        window: ``(start_day, end_day)`` half-open disruption interval.
        tolerance: Fill-rate tolerance for "recovered" (default 1 pp).

    Returns:
        Dict with ``service_level_degradation`` and ``recovery_time``.
    """
    degradation = max(0.0, float(baseline_fill_rate) - float(disruption_fill_rate))

    service = _smooth(mean_daily_service(disruption_daily), RECOVERY_SMOOTHING)
    _, end = window
    end = max(0, min(int(end), len(service)))
    post = service[end:]
    target = float(baseline_fill_rate) - tolerance
    if len(post) == 0:
        recovery_time = 0.0
    else:
        recovered = np.where(post >= target)[0]
        recovery_time = float(recovered[0]) if len(recovered) > 0 else float(len(post))

    return {
        "service_level_degradation": degradation,
        "recovery_time": recovery_time,
    }
