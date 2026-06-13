"""Economic Order Quantity (EOQ) baseline policy.

A continuous-review (s, Q) policy: when inventory position falls to or below
the reorder point ``ROP``, place a fixed order of size ``Q*`` derived from the
classical EOQ formula. Safety stock is sized for a normal-approximation
service-level target. This is the first of two classical baselines for PPO
to beat in Hypothesis 1.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from adaptive_scm.policies.base import Policy, State
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

DAYS_PER_YEAR = 365


class EOQPolicy(Policy):
    """Continuous-review EOQ policy with safety stock.

    On each call to :meth:`select_action`, recomputes the EOQ targets from the
    latest forecast (so they adapt over the horizon) and orders ``Q*`` when
    inventory position is at or below the reorder point. Uses only the
    inventory and forecast fields of :class:`State`; day-of-week and event
    flags are ignored. Stateless across periods (``reset`` is a no-op).

    Formulas (PRD Feature 5):
        - ``Q* = sqrt(2 * D * S / H)`` with ``D`` annual demand, ``S`` fixed
          order cost, ``H`` annual holding cost per unit.
        - ``ss = z * sigma_d * sqrt(L)`` where ``sigma_d`` is the daily
          forecast-error std (proxied by the forecaster's historical RMSE) and
          ``z`` is the service-level z-score from the standard normal.
        - ``ROP = mean_daily_demand * L + ss``.
    """

    def __init__(
        self,
        holding_per_unit_per_day: float,
        fixed_order_cost: float,
        lead_time: int,
        service_level: float = 0.95,
    ) -> None:
        """Configure the policy with cost and lead-time parameters.

        Validates inputs and precomputes the service-level z-score. The cost
        and lead-time parameters come from ``config/default.yaml`` (the
        ``simulation.costs`` and ``simulation.lead_time`` sections); the
        service level comes from ``config/policies/eoq.yaml``.

        Args:
            holding_per_unit_per_day: Daily holding cost per unit held.
            fixed_order_cost: Fixed cost per order placed (the ``K`` term).
            lead_time: Base (deterministic) lead time in days.
            service_level: Target cycle service level (probability of no
                stockout in a replenishment cycle). Default 0.95.

        Raises:
            ValueError: If any cost or lead-time parameter is non-positive,
                or if ``service_level`` is not in ``(0, 1)``.
        """
        if holding_per_unit_per_day <= 0:
            raise ValueError(
                f"holding_per_unit_per_day must be positive, got {holding_per_unit_per_day}"
            )
        if fixed_order_cost <= 0:
            raise ValueError(f"fixed_order_cost must be positive, got {fixed_order_cost}")
        if lead_time < 1:
            raise ValueError(f"lead_time must be >= 1, got {lead_time}")
        if not 0.0 < service_level < 1.0:
            raise ValueError(f"service_level must be in (0, 1), got {service_level}")

        self._h_daily = float(holding_per_unit_per_day)
        self._h_annual = self._h_daily * DAYS_PER_YEAR
        self._fixed_order_cost = float(fixed_order_cost)
        self._lead_time = int(lead_time)
        self._service_level = float(service_level)
        self._z = float(norm.ppf(service_level))

    def reset(self) -> None:
        """No-op reset. EOQ is stateless across decision epochs.

        Implemented to satisfy the :class:`Policy` interface; called by the
        simulator at episode start.
        """
        return None

    def select_action(self, state: State) -> int:
        """Return order quantity ``Q*`` if inventory position is at or below ``ROP``.

        Uses the lead-time-relevant slice of the forecast (first ``L`` days) to
        estimate mean daily demand, annualizes it for the EOQ numerator, and
        computes safety stock from the forecaster's historical RMSE. If
        inventory position exceeds the reorder point, returns 0 (no order).

        Args:
            state: Current :class:`State` from the simulator. Only
                ``inventory_position`` and ``forecast`` are read.

        Returns:
            Non-negative integer order quantity in units. Either 0 or the
            rounded value of ``Q*``.
        """
        mean_daily_demand = self._mean_daily_demand(state)
        q_star = self._eoq(mean_daily_demand)
        rop = self._reorder_point(state, mean_daily_demand)

        if state.inventory_position <= rop:
            return max(0, int(round(q_star)))
        return 0

    def _mean_daily_demand(self, state: State) -> float:
        """Mean forecasted daily demand over the lead-time window.

        Averages the first ``lead_time`` entries of the point forecast. Used
        in both the EOQ numerator (after annualization) and the reorder-point
        formula. Falls back to averaging the full forecast horizon if it is
        shorter than the lead time.

        Args:
            state: Current state, used for its ``forecast`` field.

        Returns:
            Mean daily demand, floored at zero to handle pathological
            negative point forecasts.
        """
        horizon = state.forecast_horizon
        window = min(self._lead_time, horizon)
        mean_d = float(state.forecast_mean[:window].mean())
        return max(0.0, mean_d)

    def _eoq(self, mean_daily_demand: float) -> float:
        """Compute ``Q* = sqrt(2 D S / H)`` from the daily demand estimate.

        Annualizes the daily demand by multiplying by ``DAYS_PER_YEAR`` and
        plugs into the standard EOQ formula. Returns ``0.0`` when expected
        demand is zero so that a non-positive square-root argument is avoided.

        Args:
            mean_daily_demand: Mean forecasted daily demand in units.

        Returns:
            EOQ in units (float, not yet rounded).
        """
        if mean_daily_demand <= 0:
            return 0.0
        annual_demand = mean_daily_demand * DAYS_PER_YEAR
        return math.sqrt(2.0 * annual_demand * self._fixed_order_cost / self._h_annual)

    def _reorder_point(self, state: State, mean_daily_demand: float) -> float:
        """Compute ``ROP = mean_daily_demand * L + ss``.

        Safety stock ``ss = z * sigma_d * sqrt(L)`` where ``sigma_d`` is the
        mean per-day forecast-error standard deviation over the lead-time
        window, read from the state's ``forecast_std`` vector (D-3.1 / D-1.1:
        ``sigma_d`` is a daily error SD, so ``sigma_d * sqrt(L)`` matches the
        textbook safety-stock form). Sourcing it from ``forecast_std`` rather
        than a single scalar lets the uncertainty vary by day and uses the same
        signal PPO sees.

        Args:
            state: Current state, used for ``forecast_std`` over the lead time.
            mean_daily_demand: Mean forecasted daily demand in units.

        Returns:
            Reorder point in units.
        """
        horizon = state.forecast_horizon
        window = min(self._lead_time, horizon)
        sigma_d = float(state.forecast_std[:window].mean())
        safety_stock = self._z * sigma_d * math.sqrt(self._lead_time)
        return mean_daily_demand * self._lead_time + safety_stock
