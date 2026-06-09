"""Abstract policy interface and shared ``State`` dataclass.

Defines the two-method contract every replenishment policy (EOQ, OrderUpTo,
PPO) must satisfy: ``select_action`` and ``reset``. The ``State`` dataclass is
the common observation passed by the simulation environment; classical policies
read named fields directly while PPO flattens it into a vector observation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from adaptive_scm.forecasting.base import ForecastOutput


@dataclass(frozen=True)
class State:
    """Decision-time state passed from the simulator to a policy.

    Holds everything a policy might need on a given day: current inventory,
    outstanding orders in the pipeline, the latest forecast, day-of-week, and
    upcoming event flags. Classical policies (EOQ, OrderUpTo) read a subset of
    these; PPO's wrapper flattens them into its observation vector.

    Attributes:
        on_hand: Current physical inventory in units.
        pipeline: Outstanding orders indexed by remaining lead time (in days).
            ``pipeline[0]`` arrives today, ``pipeline[k]`` arrives in ``k`` days.
        forecast: The latest ``ForecastOutput`` aligned to today as step 0.
        day_of_week: Integer 0..6 for the current simulation day.
        upcoming_events: Binary flags of length 7 indicating an event in each of
            the next 7 days.
    """

    on_hand: float
    pipeline: np.ndarray
    forecast: ForecastOutput
    day_of_week: int
    upcoming_events: np.ndarray

    @property
    def inventory_position(self) -> float:
        """On-hand inventory plus all outstanding orders.

        Computed as ``on_hand + sum(pipeline)``. The standard reorder-point and
        order-up-to formulas operate on inventory position rather than on-hand,
        so this is exposed as a property for clarity.

        Returns:
            Total inventory position in units.
        """
        return float(self.on_hand) + float(self.pipeline.sum())


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
