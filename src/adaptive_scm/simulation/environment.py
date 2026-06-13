"""Gymnasium inventory environment.

A single-product, single-echelon, daily-step inventory simulation. Each day the
agent (a classical policy or PPO) observes inventory, the outstanding order
pipeline, a short forward forecast, and calendar context, then chooses an order
quantity. Demand is realized as a noisy version of the forecast, stockouts are
lost sales, and reward is the negative of holding + stockout + ordering costs.

The same environment serves both worlds: PPO consumes the flat ``Box``
observation returned by ``step``/``reset``; classical policies read the
equivalent :class:`~adaptive_scm.policies.base.State` via :meth:`current_state`.
Both are projections of the same underlying simulator state, so the two policy
families face an identical problem (D-7.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from adaptive_scm.policies.base import State
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

# Forward forecast window exposed in the observation (PRD Feature 7).
FORECAST_WINDOW = 7

# Discrete action multipliers applied to mean daily demand (PRD Feature 7).
ACTION_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def state_to_observation(state: State) -> np.ndarray:
    """Flatten a :class:`State` into the env's Box observation vector.

    Concatenates on-hand, pipeline, the forecast mean and SD windows, the
    day-of-week one-hot, and the upcoming-event flags into a single float32
    array. Shared by the environment (PPO-facing observation) and the PPO
    agent (which flattens an incoming state at decision time), so the agent's
    input is identical in training and evaluation (D-9.1).

    Args:
        state: The structured state to flatten.

    Returns:
        Float32 observation array.
    """
    return np.concatenate(
        [
            np.array([state.on_hand], dtype=np.float32),
            state.pipeline.astype(np.float32),
            state.forecast_mean.astype(np.float32),
            state.forecast_std.astype(np.float32),
            state.day_of_week.astype(np.float32),
            state.upcoming_events.astype(np.float32),
        ]
    )


@dataclass(frozen=True)
class EnvConfig:
    """Cost, lead-time, and episode parameters for :class:`InventoryEnv`.

    Mirrors the ``simulation`` block of ``config/default.yaml``. Frozen so a
    single config can be shared safely across replications.

    Attributes:
        holding_cost: Holding cost per unit of on-hand inventory per day (h).
        stockout_cost: Penalty per unit of unmet demand (p).
        fixed_order_cost: Fixed cost charged when any order is placed (K).
        purchase_cost: Per-unit purchase cost (c).
        lead_time_base: Deterministic part of the lead time (days).
        lead_time_max_additional: Max extra days drawn uniformly in [0, this].
        episode_length: Number of decision days per episode.
        mean_daily_demand: Reference demand ``d_bar`` that scales the action grid.
    """

    holding_cost: float = 0.05
    stockout_cost: float = 2.00
    fixed_order_cost: float = 10.00
    purchase_cost: float = 1.00
    lead_time_base: int = 3
    lead_time_max_additional: int = 2
    episode_length: int = 28
    mean_daily_demand: float = 1.0


@dataclass
class EpisodeData:
    """Per-episode ground-truth and forecast inputs.

    Supplies the environment with the forecast the agent sees plus calendar
    context, and optionally a fixed realized-demand series. When ``demand`` is
    ``None`` the environment generates demand on each reset from the forecast
    via multiplicative lognormal noise (PRD Feature 7), so replications differ
    by their demand draws. Tests pass an explicit ``demand`` for determinism.
    Forecast arrays must extend ``FORECAST_WINDOW`` days beyond ``episode_length``
    so the final day still has a full forward window.

    Attributes:
        forecast_mean: Forecast point estimate per day, length
            >= ``episode_length + FORECAST_WINDOW``.
        forecast_std: Forecast-error SD per day, same length as ``forecast_mean``.
        day_of_week: Integer 0..6 per day, length >= ``episode_length``.
        demand: Optional fixed realized demand per day, length
            >= ``episode_length``. When ``None``, generated from the forecast.
        events: Binary event flag per day, length
            >= ``episode_length + FORECAST_WINDOW``.
    """

    forecast_mean: np.ndarray
    forecast_std: np.ndarray
    day_of_week: np.ndarray
    demand: np.ndarray | None = None
    events: np.ndarray = field(default=None)  # type: ignore[assignment]


class InventoryEnv(gym.Env):
    """Single-product inventory environment with lost sales and lead times.

    Implements the standard Gymnasium API. The observation is a flat float
    vector concatenating on-hand inventory, the pipeline (length
    ``lead_time_base + lead_time_max_additional``), the 7-day forecast mean and
    SD, a 7-day day-of-week one-hot, and 7 upcoming-event flags. Actions are
    ``Discrete(11)`` mapped to multiples of mean daily demand. Reward is the
    negative total daily cost. Orders arrive after a (possibly stochastic) lead
    time; unmet demand is lost, not backordered.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig,
        episode: EpisodeData,
        seed: int | None = None,
        episode_factory=None,
    ) -> None:
        """Build the environment from a config and an episode's data.

        Defines the action and observation spaces, stores the episode inputs,
        and seeds the internal RNG used for stochastic lead times and demand
        generation. Does not start an episode; call :meth:`reset` first.

        Args:
            config: Cost / lead-time / episode parameters.
            episode: Forecast/calendar inputs (and optionally fixed demand) for
                an episode. Used directly unless ``episode_factory`` is given.
            seed: Optional seed for the env RNG (reproducibility).
            episode_factory: Optional callable ``(rng) -> EpisodeData`` invoked
                on each :meth:`reset` to draw a fresh episode (e.g. a randomized
                start date for PPO training, D-9.2). When ``None``, the fixed
                ``episode`` is reused every reset.

        Raises:
            ValueError: If the episode arrays are too short for the configured
                episode length and forecast window.
        """
        super().__init__()
        self._cfg = config
        self._episode = episode
        self._episode_factory = episode_factory
        self._max_lead_time = config.lead_time_base + config.lead_time_max_additional
        self._validate_episode()

        self.action_space = spaces.Discrete(len(ACTION_MULTIPLIERS))
        obs_dim = 1 + self._max_lead_time + FORECAST_WINDOW + FORECAST_WINDOW + 7 + FORECAST_WINDOW
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._on_hand = 0.0
        self._pipeline = np.zeros(self._max_lead_time, dtype=float)
        self._t = 0
        # Mutable working lead time (the frozen config holds the default). The
        # lead-time disruption wrapper adjusts this rather than the config.
        self._lead_time_base = config.lead_time_base
        # Realized demand for the current episode (set on reset).
        self._demand = np.zeros(config.episode_length, dtype=float)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode and return the initial observation.

        Resets inventory to the first day's mean forecast (a neutral, non-empty
        starting stock), clears the pipeline, and rewinds the day counter.
        Re-seeds the lead-time RNG when ``seed`` is given so replications are
        reproducible.

        Args:
            seed: Optional RNG seed.
            options: Unused; present for Gymnasium API compatibility.

        Returns:
            Tuple of (flat observation, info dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._episode_factory is not None:
            self._episode = self._episode_factory(self._rng)
            self._validate_episode()
        self._demand = self._resolve_demand()
        self._t = 0
        self._on_hand = float(self._episode.forecast_mean[0])
        self._pipeline = np.zeros(self._max_lead_time, dtype=float)
        self._lead_time_base = self._cfg.lead_time_base
        return self._observation(), {}

    def _resolve_demand(self) -> np.ndarray:
        """Return the realized demand for this episode.

        Uses the episode's fixed ``demand`` array when present (deterministic
        tests / replays); otherwise generates it from the forecast via
        multiplicative lognormal noise (D-9.3): for each day,
        ``demand = round(max(0, forecast_mean * LogNormal(-s^2/2, s)))`` with
        ``s`` the per-day coefficient of variation ``forecast_std/forecast_mean``
        (floored). The ``-s^2/2`` drift keeps the multiplier's mean at 1, so
        demand is unbiased around the forecast.

        Returns:
            Realized demand array of length ``episode_length``.
        """
        n = self._cfg.episode_length
        if self._episode.demand is not None:
            return np.array(self._episode.demand[:n], dtype=float)  # copy, not view

        mean = np.asarray(self._episode.forecast_mean[:n], dtype=float)
        std = np.asarray(self._episode.forecast_std[:n], dtype=float)
        cv = np.where(mean > 1e-9, std / np.maximum(mean, 1e-9), 0.0)
        cv = np.clip(cv, 0.0, 2.0)
        noise = self._rng.lognormal(mean=-(cv**2) / 2.0, sigma=cv)
        demand = np.clip(mean * noise, 0.0, None)
        return np.round(demand)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one day: place an order, receive arrivals, meet demand, cost.

        Sequence each day: (1) the order quantity for ``action`` is scheduled to
        arrive after a stochastic lead time; (2) today's pipeline arrivals are
        added to on-hand; (3) realized demand is met from on-hand with the
        shortfall lost; (4) the daily cost is charged and converted to reward.

        Args:
            action: Index into :data:`ACTION_MULTIPLIERS`.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info). ``info``
            carries per-day ``demand``, ``order``, ``on_hand``, ``holding_cost``,
            ``stockout_cost``, ``order_cost``, and ``lost_sales``.
        """
        # Arrivals are processed before today's order is placed: an order can
        # never arrive the same day it is placed. With the order landing in slot
        # (lead - 1), this makes the realized lead time exactly `lead` days.
        arrivals = float(self._pipeline[0])
        self._pipeline = np.roll(self._pipeline, -1)
        self._pipeline[-1] = 0.0
        self._on_hand += arrivals

        order_qty = self._action_to_quantity(action)
        self._place_order(order_qty)

        demand = float(self._demand[self._t])
        sales = min(self._on_hand, demand)
        lost = demand - sales
        self._on_hand -= sales

        holding = self._cfg.holding_cost * max(self._on_hand, 0.0)
        stockout = self._cfg.stockout_cost * lost
        order_cost = self._cfg.fixed_order_cost * (1.0 if order_qty > 0 else 0.0)
        order_cost += self._cfg.purchase_cost * order_qty
        reward = -(holding + stockout + order_cost)

        info = {
            "demand": demand,
            "order": order_qty,
            "on_hand": self._on_hand,
            "holding_cost": holding,
            "stockout_cost": stockout,
            "order_cost": order_cost,
            "lost_sales": lost,
        }

        self._t += 1
        terminated = False
        truncated = self._t >= self._cfg.episode_length
        return self._observation(), reward, terminated, truncated, info

    def current_state(self) -> State:
        """Return the current decision state as a :class:`State`.

        Builds the structured state classical policies consume from the same
        internal variables that back the flat observation, so EOQ / order-up-to
        and PPO face the identical situation (D-7.1). Called by the runner each
        day before invoking a classical policy.

        Returns:
            The current :class:`State`.
        """
        lo = self._t
        hi = self._t + FORECAST_WINDOW
        dow = np.zeros(7, dtype=np.int8)
        dow[int(self._episode.day_of_week[min(self._t, len(self._episode.day_of_week) - 1)])] = 1
        return State(
            on_hand=self._on_hand,
            pipeline=self._pipeline.copy(),
            forecast_mean=self._episode.forecast_mean[lo:hi].astype(float),
            forecast_std=self._episode.forecast_std[lo:hi].astype(float),
            day_of_week=dow,
            upcoming_events=self._events_window(),
            time_index=self._t,
        )

    def _action_to_quantity(self, action: int) -> float:
        """Map a discrete action index to an order quantity.

        Multiplies the configured mean daily demand by the action's multiplier
        from :data:`ACTION_MULTIPLIERS`.

        Args:
            action: Index into the multiplier grid.

        Returns:
            Order quantity in units (float).

        Raises:
            ValueError: If ``action`` is outside the grid.
        """
        if not 0 <= action < len(ACTION_MULTIPLIERS):
            raise ValueError(f"action {action} outside [0, {len(ACTION_MULTIPLIERS)})")
        return ACTION_MULTIPLIERS[action] * self._cfg.mean_daily_demand

    def order_units(self, units: float) -> int:
        """Map a desired unit quantity to the nearest discrete action.

        Lets classical policies, which compute a continuous order quantity, drive
        the discrete-action environment: picks the action whose resulting
        quantity is closest to ``units`` (D-7.2).

        Args:
            units: Desired order quantity in units.

        Returns:
            The action index in ``[0, len(ACTION_MULTIPLIERS))``.
        """
        quantities = np.array(ACTION_MULTIPLIERS) * self._cfg.mean_daily_demand
        return int(np.argmin(np.abs(quantities - units)))

    def _place_order(self, order_qty: float) -> None:
        """Schedule an order to arrive after a stochastic lead time.

        Draws an extra delay uniformly in ``[0, lead_time_max_additional]`` and
        adds the order to the pipeline slot ``base + extra - 1`` (so a lead time
        of L lands in the slot that rolls to on-hand after L days). Mutates the
        pipeline in place.

        Args:
            order_qty: Units ordered today (zero means no order).
        """
        if order_qty <= 0:
            return
        extra = int(self._rng.integers(0, self._cfg.lead_time_max_additional + 1))
        lead = self._lead_time_base + extra
        slot = min(lead - 1, self._max_lead_time - 1)
        self._pipeline[slot] += order_qty

    def _events_window(self) -> np.ndarray:
        """Return the 7-day upcoming-event flag vector from the current day.

        Slices the episode's event array; returns zeros when no event data was
        supplied. Padded with zeros if the window runs past the array end.

        Returns:
            Binary array of length :data:`FORECAST_WINDOW`.
        """
        if self._episode.events is None:
            return np.zeros(FORECAST_WINDOW, dtype=np.int8)
        lo = self._t
        hi = self._t + FORECAST_WINDOW
        window = self._episode.events[lo:hi]
        if len(window) < FORECAST_WINDOW:
            window = np.pad(window, (0, FORECAST_WINDOW - len(window)))
        return window.astype(np.int8)

    def _observation(self) -> np.ndarray:
        """Flatten the current state into the Box observation vector.

        Delegates to :func:`state_to_observation` so the PPO-facing observation
        is byte-identical to what a policy would flatten from
        :meth:`current_state` — guaranteeing PPO sees the same inputs at
        training and evaluation time.

        Returns:
            Float32 observation array.
        """
        return state_to_observation(self.current_state())

    def _validate_episode(self) -> None:
        """Check the episode arrays are long enough for the configured horizon.

        The demand and day-of-week arrays must cover the episode; the forecast
        and event arrays must additionally cover the trailing forecast window.
        Called once from ``__init__``.

        Raises:
            ValueError: If any array is too short.
        """
        n = self._cfg.episode_length
        if self._episode.demand is not None and len(self._episode.demand) < n:
            raise ValueError(f"demand array shorter than episode_length ({n})")
        if len(self._episode.day_of_week) < n:
            raise ValueError(f"day_of_week array shorter than episode_length ({n})")
        need = n + FORECAST_WINDOW
        if len(self._episode.forecast_mean) < need or len(self._episode.forecast_std) < need:
            raise ValueError(f"forecast arrays must cover episode_length + window ({need} days)")
