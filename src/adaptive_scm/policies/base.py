"""Abstract base class and state container for inventory replenishment policies.

Defines the `Policy` interface that EOQ, OrderUpTo, and PPO all implement,
plus the `State` dataclass passed to `select_action`. This module is imported
by every concrete policy and by the simulation environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class State:
    """Observable state passed to a policy at each decision epoch.

    Different policies consume different subsets: EOQ uses inventory_position
    and forecast_mean; OrderUpTo additionally uses forecast_std; PPO uses
    everything (the full vector is flattened inside the agent).

    Attributes:
        on_hand: Current on-hand inventory in units.
        pipeline: Outstanding orders by remaining lead time. Shape
            (max_lead_time,). pipeline[i] is units arriving in i+1 days.
        forecast_mean: Point forecast for the next horizon days, shape
            (forecast_horizon,). Typically the 7-day slice used by PPO.
        forecast_std: Forecast uncertainty (std) over the same horizon,
            shape (forecast_horizon,).
        day_of_week: One-hot encoding of current day-of-week, shape (7,).
        upcoming_events: Binary event flags for the next 7 days, shape (7,).
        time_index: Integer step within the current episode (0-indexed).
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
        """Inventory position = on-hand + sum of pipeline orders.

        Standard textbook definition used by classical replenishment policies
        (EOQ, OrderUpTo) when comparing against reorder points / targets.
        """
        return float(self.on_hand + self.pipeline.sum())


class Policy(ABC):
    """Abstract interface for all inventory replenishment policies.

    Every concrete policy (EOQ, OrderUpTo, PPO) implements this two-method
    contract. The simulation environment calls `select_action` once per day
    and `reset` at the start of each episode/replication.
    """

    @abstractmethod
    def select_action(self, state: State) -> int:
        """Choose the order quantity for the current decision epoch.

        For classical policies this is a deterministic function of the state;
        for PPO it samples from a learned stochastic policy (or returns the
        deterministic mode at evaluation time).

        Args:
            state: Current observable state.

        Returns:
            Order quantity in units. Must be non-negative integer.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any episode-internal state at the start of a new replication.

        Stateless policies may implement this as a no-op. Stateful policies
        (e.g., PPO with recurrent components, or order-up-to with a running
        forecast cache) clear their per-episode state here. Called by the
        runner before each replication.
        """
