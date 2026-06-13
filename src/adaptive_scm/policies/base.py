"""Abstract policy interface and shared ``State`` dataclass.

Defines the two-method contract every replenishment policy (EOQ, OrderUpTo,
PPO) must satisfy: ``select_action`` and ``reset``. The ``State`` dataclass is
the simulator's native per-day observation; it mirrors the PRD's PPO state
vector so the PPO wrapper can flatten it almost directly, while classical
policies read the named fields they need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class State:
    """Decision-time state passed from the simulator to a policy.

    Holds the raw facts a policy needs on a given simulation day: inventory,
    outstanding pipeline orders, the forward forecast as separate mean and
    uncertainty vectors, calendar position, and upcoming events. The field
    layout matches the PRD's PPO observation (Feature 7) so the PPO wrapper can
    concatenate the fields into its vector with minimal transformation; EOQ and
    OrderUpTo read the subset they need (inventory position and the forecast).

    Attributes:
        on_hand: Current physical inventory in units.
        pipeline: Outstanding orders indexed by remaining lead time (days).
            ``pipeline[k]`` arrives in ``k`` days (``pipeline[0]`` arrives today).
        forecast_mean: Point forecast of demand for the next ``H`` days
            (``H`` is the forecast window the simulator exposes, default 7).
        forecast_std: Per-day forecast-error standard deviation over the same
            window; the uncertainty signal for both PPO and classical safety stock.
        day_of_week: One-hot vector of length 7 for the current day.
        upcoming_events: Binary flags of length 7 marking an event on each of
            the next 7 days.
        time_index: Integer day index within the episode (0-based).
    """

    on_hand: float
    pipeline: np.ndarray
    forecast_mean: np.ndarray
    forecast_std: np.ndarray
    day_of_week: np.ndarray
    upcoming_events: np.ndarray
    time_index: int

    @property
    def inventory_position(self) -> float:
        """On-hand inventory plus all outstanding pipeline orders.

        Computed as ``on_hand + sum(pipeline)``. Reorder-point and order-up-to
        formulas operate on inventory position rather than on-hand alone, so it
        is exposed as a property.

        Returns:
            Total inventory position in units.
        """
        return float(self.on_hand) + float(self.pipeline.sum())

    @property
    def forecast_horizon(self) -> int:
        """Length of the forecast window carried in the state.

        Returns:
            Number of days in ``forecast_mean`` / ``forecast_std``.
        """
        return int(self.forecast_mean.shape[0])


class Policy(ABC):
    """Abstract base class for all replenishment policies.

    Concrete policies are interchangeable from the simulator's perspective. The
    simulator calls ``reset`` at episode start and ``select_action`` once per
    decision epoch.
    """

    @abstractmethod
    def select_action(self, state: State) -> int:
        """Return the integer order quantity for the current decision epoch.

        Args:
            state: Current ``State`` from the simulator.

        Returns:
            Non-negative integer order quantity in units.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any per-episode internal state.

        Stateless policies may implement this as a no-op. Called by the
        simulator at the start of every episode.
        """
