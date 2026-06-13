"""Fast unit tests for the PPO agent (PRD Feature 9).

These train PPO for only a few hundred steps with a tiny rollout so they run in
seconds; the full 5000-step training gate lives in
``tests/integration/test_pipeline.py``. Marked ``ppo`` so the runner keeps these
torch-backed tests in the same process group as TFT and away from xgboost
(macOS libomp clash, D-4.7).
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_scm.policies import PPOAgent, PPOHyperparams
from adaptive_scm.policies.base import Policy
from adaptive_scm.simulation import EnvConfig, InventoryEnv, make_training_episode_factory

pytestmark = pytest.mark.ppo

_FAST_HP = PPOHyperparams(n_steps=128, batch_size=64, n_epochs=2)


def _series(length: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic weekly-seasonal sales series and its day-of-week.

    Args:
        length: Number of days.

    Returns:
        Tuple ``(sales, day_of_week)``.
    """
    rng = np.random.default_rng(0)
    days = np.arange(length)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 1, length), 1, None)
    return sales, (days % 7)


def _training_env(episode_length: int = 100) -> tuple[InventoryEnv, float]:
    """Build a training env with a randomized-start episode factory.

    Args:
        episode_length: Episode length in days.

    Returns:
        Tuple ``(env, mean_daily_demand)``.
    """
    sales, dow = _series()
    d_bar = float(sales.mean())
    cfg = EnvConfig(episode_length=episode_length, mean_daily_demand=d_bar)
    factory = make_training_episode_factory(
        sales, dow, historical_rmse=2.0, episode_length=episode_length
    )
    env = InventoryEnv(cfg, factory(np.random.default_rng(0)), seed=0, episode_factory=factory)
    return env, d_bar


@pytest.fixture(scope="module")
def trained_agent() -> tuple[PPOAgent, InventoryEnv]:
    """A PPO agent trained for a few hundred steps, shared across tests.

    Returns:
        Tuple ``(agent, env)`` where ``env`` is the training environment.
    """
    env, d_bar = _training_env()
    agent = PPOAgent(mean_daily_demand=d_bar, hyperparams=_FAST_HP, seed=0)
    agent.train(env, total_timesteps=256)
    return agent, env


class TestConstruction:
    def test_rejects_non_positive_demand(self):
        with pytest.raises(ValueError, match="mean_daily_demand"):
            PPOAgent(mean_daily_demand=0.0)

    def test_select_action_before_train_raises(self):
        from adaptive_scm.policies.base import State

        agent = PPOAgent(mean_daily_demand=10.0)
        dummy = State(
            on_hand=0.0,
            pipeline=np.zeros(5),
            forecast_mean=np.zeros(7),
            forecast_std=np.zeros(7),
            day_of_week=np.eye(7)[0],
            upcoming_events=np.zeros(7),
            time_index=0,
        )
        with pytest.raises(RuntimeError, match="not trained"):
            agent.select_action(dummy)

    def test_is_a_policy(self, trained_agent):
        agent, _ = trained_agent
        assert isinstance(agent, Policy)


class TestActions:
    def test_select_action_returns_nonneg_int_units(self, trained_agent):
        agent, env = trained_agent
        env.reset(seed=7)
        units = agent.select_action(env.current_state())
        assert isinstance(units, int)
        assert units >= 0

    def test_action_index_in_range(self, trained_agent):
        agent, env = trained_agent
        env.reset(seed=7)
        idx = agent.action_index(env.current_state())
        assert 0 <= idx < env.action_space.n

    def test_units_consistent_with_index(self, trained_agent):
        # The returned units must equal the chosen action's multiplier * d_bar,
        # so the runner's order_units round-trips back to the same index.
        agent, env = trained_agent
        env.reset(seed=3)
        state = env.current_state()
        units = agent.select_action(state)
        idx = agent.action_index(state)
        assert env.order_units(units) == idx

    def test_deterministic_actions(self, trained_agent):
        agent, env = trained_agent
        env.reset(seed=9)
        state = env.current_state()
        assert agent.select_action(state) == agent.select_action(state)

    def test_reset_is_noop(self, trained_agent):
        agent, _ = trained_agent
        assert agent.reset() is None


class TestSaveLoad:
    def test_round_trip_preserves_actions(self, trained_agent, tmp_path):
        agent, env = trained_agent
        path = tmp_path / "ppo_agent"
        agent.save(path)
        assert (tmp_path / "ppo_agent.zip").exists()

        loaded = PPOAgent.load(tmp_path / "ppo_agent.zip", mean_daily_demand=10.0)
        env.reset(seed=11)
        state = env.current_state()
        # Same observation -> same action index (the model is restored exactly).
        assert loaded.action_index(state) == agent.action_index(state)

    def test_save_before_train_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not trained"):
            PPOAgent(mean_daily_demand=10.0).save(tmp_path / "x")


class TestEpisodeFactory:
    def test_randomized_episodes_differ(self):
        env, _ = _training_env()
        o1, _ = env.reset(seed=1)
        o2, _ = env.reset(seed=2)
        # Different start dates -> different initial observations.
        assert not np.array_equal(o1, o2)
