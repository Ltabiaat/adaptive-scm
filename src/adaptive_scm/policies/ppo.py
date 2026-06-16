"""Proximal Policy Optimization inventory agent.

Wraps Stable Baselines3's PPO behind the
:class:`~adaptive_scm.policies.base.Policy` interface so the learned agent is
interchangeable with EOQ and OrderUpTo in the simulation and experiment runner.
The agent trains on an :class:`~adaptive_scm.simulation.environment.InventoryEnv`
whose state already carries a frozen forecaster's forecast; PPO never retrains
the forecaster (PRD Feature 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from adaptive_scm.policies.base import Policy, State
from adaptive_scm.utils.device import resolve_device
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class PPOHyperparams:
    """PPO training hyperparameters (PRD Feature 9 defaults).

    Mirrors the ``policies.ppo`` config block. Frozen so a single set can be
    shared across runs.

    Attributes:
        clip_range: PPO clipping parameter (epsilon).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        learning_rate: Optimizer learning rate.
        batch_size: Minibatch size.
        n_epochs: Optimization epochs per rollout.
        n_steps: Steps collected per rollout before each update.
        net_arch: Hidden-layer sizes of the MLP policy/value networks.
    """

    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 4
    n_steps: int = 2048
    net_arch: tuple[int, ...] = (64, 64)


class PPOAgent(Policy):
    """Learned replenishment policy backed by Stable Baselines3 PPO.

    Trains an ``MlpPolicy`` PPO agent on the inventory environment, then serves
    decisions through :meth:`select_action`: the incoming :class:`State` is
    flattened with the env's shared :func:`state_to_observation` (so inputs match
    training exactly, D-9.1) and the policy is queried deterministically. Like
    the classical policies, :meth:`select_action` returns an order **quantity in
    units** (the chosen action's multiplier times mean daily demand), which the
    runner maps back to the discrete action via ``env.order_units`` (D-7.2 /
    D-9.4). Stateless between steps (``reset`` is a no-op for the MLP policy).
    """

    def __init__(
        self,
        mean_daily_demand: float,
        hyperparams: PPOHyperparams | None = None,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        """Configure the agent (no training yet).

        Args:
            mean_daily_demand: The env's ``d_bar``; scales the action grid so the
                agent's unit output matches the env's action discretization.
            hyperparams: PPO hyperparameters; defaults to the PRD values.
            seed: Seed passed to PPO for reproducibility.
            device: Compute device preference; resolved via
                ``utils.device.resolve_device`` to ``"cuda"`` or ``"cpu"`` (never
                MPS). ``"auto"`` uses CUDA when available, else CPU.

        Raises:
            ValueError: If ``mean_daily_demand`` is non-positive.
        """
        if mean_daily_demand <= 0:
            raise ValueError(f"mean_daily_demand must be positive, got {mean_daily_demand}")
        self._d_bar = float(mean_daily_demand)
        self._hp = hyperparams or PPOHyperparams()
        self._seed = int(seed)
        self._device = resolve_device(device)
        self._model = None  # set by train() or load()

    def train(self, env, total_timesteps: int, normalize_reward: bool = False):
        """Train the PPO agent on an inventory environment.

        Builds an SB3 ``PPO("MlpPolicy", ...)`` with the configured
        hyperparameters and a ``[net_arch]`` tanh network, optionally wrapping
        the env in ``VecNormalize`` for reward scaling (PRD open question on
        reward magnitudes), and runs ``learn`` for ``total_timesteps`` steps.
        The env's state already contains the frozen forecast, so the forecaster
        is never touched here.

        Args:
            env: A Gymnasium ``InventoryEnv`` (or wrapped variant).
            total_timesteps: Environment steps to train for.
            normalize_reward: If True, wrap in ``VecNormalize`` (reward only).

        Returns:
            ``self``, to allow chaining.
        """
        import torch.nn as nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        vec_env = DummyVecEnv([lambda: env])
        if normalize_reward:
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

        self._model = PPO(
            "MlpPolicy",
            vec_env,
            clip_range=self._hp.clip_range,
            gamma=self._hp.gamma,
            gae_lambda=self._hp.gae_lambda,
            learning_rate=self._hp.learning_rate,
            batch_size=self._hp.batch_size,
            n_epochs=self._hp.n_epochs,
            n_steps=self._hp.n_steps,
            policy_kwargs=dict(net_arch=list(self._hp.net_arch), activation_fn=nn.Tanh),
            seed=self._seed,
            device=self._device,
            verbose=0,
        )
        self._model.learn(total_timesteps=total_timesteps)
        _LOG.info(
            "ppo_trained",
            total_timesteps=total_timesteps,
            n_steps=self._hp.n_steps,
            normalize_reward=normalize_reward,
            device=self._device,
        )
        return self

    def select_action(self, state: State) -> int:
        """Return the order quantity (units) for the current state.

        Flattens ``state`` with :func:`state_to_observation`, queries the policy
        deterministically for an action index, and converts that index to an
        order quantity (``multiplier * d_bar``). Returning units keeps the
        interface uniform with the classical policies (D-9.4); the runner maps
        the value back to the discrete action.

        Args:
            state: Current decision state.

        Returns:
            Non-negative integer order quantity in units.

        Raises:
            RuntimeError: If called before :meth:`train` or :meth:`load`.
        """
        from adaptive_scm.simulation.environment import ACTION_MULTIPLIERS, state_to_observation

        self._require_trained()
        obs = state_to_observation(state)
        action, _ = self._model.predict(obs, deterministic=True)
        quantity = ACTION_MULTIPLIERS[int(action)] * self._d_bar
        return max(0, int(round(quantity)))

    def action_index(self, state: State) -> int:
        """Return the raw discrete action index for a state.

        Convenience for callers (and tests) that want the policy's chosen
        action directly rather than the unit quantity. Deterministic.

        Args:
            state: Current decision state.

        Returns:
            Action index in ``[0, len(ACTION_MULTIPLIERS))``.

        Raises:
            RuntimeError: If called before :meth:`train` or :meth:`load`.
        """
        from adaptive_scm.simulation.environment import state_to_observation

        self._require_trained()
        action, _ = self._model.predict(state_to_observation(state), deterministic=True)
        return int(action)

    def reset(self) -> None:
        """No-op reset. The MLP policy carries no per-episode state.

        Implemented to satisfy the :class:`Policy` interface; called by the
        simulator at episode start.
        """
        return None

    def save(self, path: Path) -> None:
        """Persist the trained agent to a ``.zip`` via SB3.

        Args:
            path: Destination path (SB3 appends ``.zip`` if absent).

        Raises:
            RuntimeError: If called before training/loading.
        """
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        _LOG.info("ppo_saved", path=str(path))

    @classmethod
    def load(cls, path: Path, mean_daily_demand: float) -> "PPOAgent":
        """Load a trained agent saved by :meth:`save`.

        Args:
            path: Path passed to a prior :meth:`save` call.
            mean_daily_demand: The env ``d_bar`` for the action-to-units mapping
                (not stored in the SB3 zip).

        Returns:
            A ready-to-serve :class:`PPOAgent`.
        """
        from stable_baselines3 import PPO

        instance = cls(mean_daily_demand=mean_daily_demand)
        instance._model = PPO.load(str(path), device=instance._device)
        _LOG.info("ppo_loaded", path=str(path), device=instance._device)
        return instance

    def _require_trained(self) -> None:
        """Raise if the agent has no model yet.

        Raises:
            RuntimeError: If neither :meth:`train` nor :meth:`load` has run.
        """
        if self._model is None:
            raise RuntimeError("PPO agent is not trained; call train() or load() first")
