"""Disruption scenario wrappers.

Two ``gymnasium.Wrapper`` subclasses that perturb :class:`InventoryEnv` over a
configurable window without touching the core environment code (PRD Feature 8):
a demand spike (realized demand multiplied up) and a lead-time disruption (base
lead time multiplied up). The undisrupted baseline is simply the bare env.
"""

from __future__ import annotations

import gymnasium as gym

from adaptive_scm.simulation.environment import InventoryEnv
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


class DemandSpikeWrapper(gym.Wrapper):
    """Multiply realized demand by a factor over a fixed window.

    Wraps an :class:`InventoryEnv` and scales the underlying episode's demand
    array in place over ``[start_day, start_day + duration)``. Because the scale
    is applied to the env's ground-truth demand (not the forecast), the agent is
    not warned of the spike, which is the point of the stress test.

    The wrapper is idempotent across resets: it records the original demand and
    re-applies the scaling from the pristine copy, so repeated episodes see the
    same disruption.
    """

    def __init__(
        self,
        env: InventoryEnv,
        multiplier: float = 1.5,
        start_day: int = 7,
        duration: int = 14,
    ) -> None:
        """Configure the demand-spike window.

        Args:
            env: The inventory environment to wrap.
            multiplier: Factor applied to demand within the window.
            start_day: First day of the spike (inclusive).
            duration: Number of days the spike lasts.

        Raises:
            ValueError: If ``multiplier`` is non-positive or the window is invalid.
        """
        super().__init__(env)
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        if start_day < 0 or duration < 1:
            raise ValueError("start_day must be >= 0 and duration >= 1")
        self._multiplier = float(multiplier)
        self._start = int(start_day)
        self._end = int(start_day) + int(duration)
        self._original_demand = env.unwrapped._episode.demand.copy()
        self._apply()

    def _apply(self) -> None:
        """Scale the env's demand array within the window from the pristine copy.

        Restores the original demand first, then multiplies the window slice, so
        the disruption is exactly applied once regardless of how many times
        ``reset`` has been called. Mutates the wrapped env's episode in place.
        """
        demand = self._original_demand.copy()
        end = min(self._end, len(demand))
        demand[self._start : end] *= self._multiplier
        self.env.unwrapped._episode.demand = demand

    def reset(self, **kwargs):
        """Re-apply the spike and delegate to the wrapped env's reset.

        Returns:
            The wrapped env's ``(observation, info)``.
        """
        self._apply()
        return self.env.reset(**kwargs)


class LeadTimeDisruptionWrapper(gym.Wrapper):
    """Multiply the base lead time over a fixed window.

    Wraps an :class:`InventoryEnv` and, while the simulation day is within
    ``[start_day, start_day + duration)``, temporarily scales the env's base
    lead time before each ``step`` (then restores it). Orders placed during the
    window therefore take longer to arrive. The pipeline length is sized for the
    maximum possible lead time, so the disrupted lead time still fits.
    """

    def __init__(
        self,
        env: InventoryEnv,
        multiplier: float = 2.0,
        start_day: int = 7,
        duration: int = 14,
    ) -> None:
        """Configure the lead-time disruption window.

        Args:
            env: The inventory environment to wrap.
            multiplier: Factor applied to the base lead time within the window.
            start_day: First day of the disruption (inclusive).
            duration: Number of days the disruption lasts.

        Raises:
            ValueError: If ``multiplier < 1`` or the window is invalid.
        """
        super().__init__(env)
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        if start_day < 0 or duration < 1:
            raise ValueError("start_day must be >= 0 and duration >= 1")
        self._multiplier = float(multiplier)
        self._start = int(start_day)
        self._end = int(start_day) + int(duration)
        self._base_lead = env.unwrapped._cfg.lead_time_base

    def step(self, action):
        """Temporarily inflate the working lead time during the window, then step.

        Reads the env's current day; if inside the window, sets the env's
        mutable ``_lead_time_base`` to ``round(base * multiplier)`` (capped so the
        arrival slot fits the pipeline) for the duration of this step, then
        restores it in a ``finally`` so an exception cannot leave it mutated.

        Args:
            action: The action to pass through.

        Returns:
            The wrapped env's ``step`` return tuple.
        """
        env = self.env.unwrapped
        in_window = self._start <= env._t < self._end
        original = env._lead_time_base
        if in_window:
            inflated = min(round(self._base_lead * self._multiplier), env._max_lead_time)
            env._lead_time_base = int(inflated)
        try:
            return self.env.step(action)
        finally:
            if in_window:
                env._lead_time_base = original
