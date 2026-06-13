"""Episode construction for the inventory simulation.

Helpers that turn a preprocessed sales series and a frozen forecaster into
:class:`~adaptive_scm.simulation.environment.EpisodeData` instances. The PPO
training loop uses a randomized-start-date factory; evaluation uses a single
fixed window. The forecaster is queried once and its outputs are treated as
fixed inputs (the agent is never allowed to retrain it).
"""

from __future__ import annotations

import numpy as np

from adaptive_scm.simulation.environment import FORECAST_WINDOW, EpisodeData
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


def build_forecast_arrays(
    sales: np.ndarray,
    historical_rmse: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-day forecast mean and SD arrays over a series.

    For the simulation the forecast "signal" is the realized sales level and the
    uncertainty is the forecaster's validation RMSE held constant per day. The
    environment then generates stochastic demand around this signal (D-9.3), so
    the agent faces a noisy realization of a forecast of known quality. This is
    the frozen-forecaster contract: the forecaster's accuracy (RMSE) enters the
    simulation, but it is never retrained.

    Args:
        sales: Daily sales/forecast level, length ``L``.
        historical_rmse: The forecaster's validation RMSE (uncertainty level).

    Returns:
        Tuple ``(forecast_mean, forecast_std)`` each of length ``L``.
    """
    forecast_mean = np.asarray(sales, dtype=float)
    forecast_std = np.full(len(sales), float(historical_rmse), dtype=float)
    return forecast_mean, forecast_std


def make_training_episode_factory(
    sales: np.ndarray,
    day_of_week: np.ndarray,
    historical_rmse: float,
    episode_length: int,
):
    """Return a factory that draws random fixed-length training episodes.

    The returned callable, given an RNG, samples a random start index into the
    series and slices an ``episode_length``-day window (plus the trailing
    forecast window), producing an :class:`EpisodeData` with no fixed demand so
    the environment generates stochastic demand each reset. This implements the
    PRD's "episodes randomized across the training period" requirement (D-9.2).

    Args:
        sales: Full training-period sales/forecast level.
        day_of_week: Day-of-week integers aligned to ``sales``.
        historical_rmse: Forecaster validation RMSE for the uncertainty level.
        episode_length: Episode length in days.

    Returns:
        A callable ``(rng) -> EpisodeData``.

    Raises:
        ValueError: If the series is too short for one episode plus the window.
    """
    need = episode_length + FORECAST_WINDOW
    if len(sales) < need:
        raise ValueError(f"series ({len(sales)}) shorter than episode+window ({need})")
    max_start = len(sales) - need

    def factory(rng: np.random.Generator) -> EpisodeData:
        start = int(rng.integers(0, max_start + 1))
        window = slice(start, start + need)
        f_mean, f_std = build_forecast_arrays(sales[window], historical_rmse)
        return EpisodeData(
            forecast_mean=f_mean,
            forecast_std=f_std,
            day_of_week=day_of_week[window][:episode_length],
            demand=None,  # generated stochastically by the env
            events=np.zeros(need, dtype=np.int8),
        )

    return factory


def build_eval_episode(
    sales: np.ndarray,
    day_of_week: np.ndarray,
    historical_rmse: float,
    episode_length: int,
) -> EpisodeData:
    """Build a single fixed evaluation episode from the tail of a series.

    Slices the last ``episode_length + FORECAST_WINDOW`` days so the evaluation
    window aligns with the most recent data (the test horizon). Demand is left
    unset so each replication generates its own demand draw.

    Args:
        sales: Series whose tail is the evaluation window.
        day_of_week: Day-of-week integers aligned to ``sales``.
        historical_rmse: Forecaster validation RMSE for the uncertainty level.
        episode_length: Episode length in days.

    Returns:
        An :class:`EpisodeData` for evaluation.

    Raises:
        ValueError: If the series is too short.
    """
    need = episode_length + FORECAST_WINDOW
    if len(sales) < need:
        raise ValueError(f"series ({len(sales)}) shorter than episode+window ({need})")
    tail = slice(len(sales) - need, len(sales))
    f_mean, f_std = build_forecast_arrays(sales[tail], historical_rmse)
    return EpisodeData(
        forecast_mean=f_mean,
        forecast_std=f_std,
        day_of_week=day_of_week[tail][:episode_length],
        demand=None,
        events=np.zeros(need, dtype=np.int8),
    )
