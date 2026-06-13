"""Forecast-driven order-up-to (base-stock) policy.

A periodic-review dynamic base-stock policy: each review epoch it raises the
inventory position up to a target level ``S_t`` computed from the forecast over
the protection interval (review period + lead time) plus safety stock. It
consumes exactly the same forecast information as the PPO agent, which is what
makes the order-up-to-vs-PPO comparison isolate the value of *learning* rather
than the value of *information* (the thesis's load-bearing experiment).
"""

from __future__ import annotations

import math

from scipy.stats import norm

from adaptive_scm.policies.base import Policy, State
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


class OrderUpToPolicy(Policy):
    """Dynamic base-stock policy driven by the forecast.

    On each call to :meth:`select_action`, computes the protection-interval
    target ``S_t = sum(forecast_mean over R+L) + z * sigma_{R+L}`` and orders
    ``max(0, S_t - inventory_position)``. Reads only the inventory and forecast
    fields of :class:`State`; like EOQ it ignores day-of-week and events.
    Stateless across periods (``reset`` is a no-op).

    Formulas (PRD Feature 6):
        - ``S_t = sum(forecast_mean[:R+L]) + z * sigma_{R+L}``
        - ``sigma_{R+L} = sqrt(R+L) * mean(forecast_std[:R+L])`` (D-3.1 / D-1.1:
          ``forecast_std`` holds the daily error SD, so the protection-interval
          SD scales by ``sqrt(R+L)`` for independent daily errors).
        - ``a_t = max(0, S_t - inventory_position)``.
    """

    def __init__(
        self,
        lead_time: int,
        review_period: int = 1,
        service_level: float = 0.95,
    ) -> None:
        """Configure the protection interval and service level.

        Validates inputs and precomputes the service-level z-score. ``lead_time``
        comes from ``config/default.yaml`` (``simulation.lead_time.base``);
        ``review_period`` and ``service_level`` from
        ``config/policies/order_up_to.yaml``.

        Args:
            lead_time: Base (deterministic) lead time in days.
            review_period: Days between review epochs (R). Default 1.
            service_level: Target cycle service level. Default 0.95.

        Raises:
            ValueError: If ``lead_time`` or ``review_period`` is below 1, or
                ``service_level`` is not in ``(0, 1)``.
        """
        if lead_time < 1:
            raise ValueError(f"lead_time must be >= 1, got {lead_time}")
        if review_period < 1:
            raise ValueError(f"review_period must be >= 1, got {review_period}")
        if not 0.0 < service_level < 1.0:
            raise ValueError(f"service_level must be in (0, 1), got {service_level}")

        self._lead_time = int(lead_time)
        self._review_period = int(review_period)
        self._service_level = float(service_level)
        self._z = float(norm.ppf(service_level))

    @property
    def protection_interval(self) -> int:
        """Length of the protection interval ``R + L`` in days.

        Returns:
            Review period plus lead time.
        """
        return self._review_period + self._lead_time

    def reset(self) -> None:
        """No-op reset. Order-up-to is stateless across decision epochs.

        Implemented to satisfy the :class:`Policy` interface; called by the
        simulator at episode start.
        """
        return None

    def select_action(self, state: State) -> int:
        """Return ``max(0, S_t - inventory_position)`` as the order quantity.

        Computes the order-up-to target from the forecast over the protection
        interval plus safety stock, then orders the gap to the current
        inventory position. Returns an integer (rounded, floored at zero).

        Args:
            state: Current :class:`State`. Only ``inventory_position`` and the
                forecast vectors are read.

        Returns:
            Non-negative integer order quantity in units.
        """
        target = self._target_level(state)
        order = target - state.inventory_position
        return max(0, int(round(order)))

    def _target_level(self, state: State) -> float:
        """Compute the order-up-to level ``S_t``.

        Sums the mean forecast over the protection interval and adds safety
        stock ``z * sigma_{R+L}``. The window is clamped to the forecast
        horizon carried in the state when ``R + L`` exceeds it.

        Args:
            state: Current state, for its forecast vectors.

        Returns:
            Target inventory level in units.
        """
        horizon = state.forecast_horizon
        window = min(self.protection_interval, horizon)
        expected_demand = float(state.forecast_mean[:window].sum())
        sigma_window = float(state.forecast_std[:window].mean())
        safety_stock = self._z * sigma_window * math.sqrt(self.protection_interval)
        return expected_demand + safety_stock
