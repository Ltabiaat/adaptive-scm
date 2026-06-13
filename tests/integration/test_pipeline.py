"""Integration tests for the end-to-end pipeline.

Slower tests that exercise full components together. ``test_ppo_training`` is the
PRD Feature 9 acceptance gate, run with a reduced step budget for CI speed. Not
part of the fast unit suite; run explicitly with
``uv run pytest tests/integration/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_scm.policies import PPOAgent, PPOHyperparams
from adaptive_scm.simulation import EnvConfig, InventoryEnv, make_training_episode_factory

pytestmark = pytest.mark.ppo


def _training_env(episode_length: int = 180):
    """Build a realistic training env with randomized-start episodes.

    Args:
        episode_length: Episode length in days.

    Returns:
        Tuple ``(env, mean_daily_demand)``.
    """
    rng = np.random.default_rng(0)
    length = 600
    days = np.arange(length)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 1.5, length), 1, None)
    dow = days % 7
    d_bar = float(sales.mean())
    cfg = EnvConfig(episode_length=episode_length, mean_daily_demand=d_bar)
    factory = make_training_episode_factory(
        sales, dow, historical_rmse=2.5, episode_length=episode_length
    )
    env = InventoryEnv(cfg, factory(rng), seed=0, episode_factory=factory)
    return env, d_bar


def test_ppo_training(tmp_path):
    """PRD Feature 9 gate: PPO trains on a reduced budget and serves valid actions.

    Trains for 5000 steps (the PRD's CI-speed budget), confirms the agent then
    produces in-range actions for fresh states, and that a saved agent reloads
    and reproduces the same action — the three Feature 9 acceptance criteria
    that are testable in CI.
    """
    env, d_bar = _training_env()
    hp = PPOHyperparams(n_steps=512, batch_size=64, n_epochs=4)
    agent = PPOAgent(mean_daily_demand=d_bar, hyperparams=hp, seed=0)
    agent.train(env, total_timesteps=5000)

    # Produces valid actions for any state.
    for seed in (1, 2, 3):
        env.reset(seed=seed)
        state = env.current_state()
        idx = agent.action_index(state)
        assert 0 <= idx < env.action_space.n
        assert isinstance(agent.select_action(state), int)

    # A trained agent can be saved and loaded, reproducing actions.
    path = tmp_path / "ppo_arima"
    agent.save(path)
    assert (tmp_path / "ppo_arima.zip").exists()
    loaded = PPOAgent.load(tmp_path / "ppo_arima.zip", mean_daily_demand=d_bar)
    env.reset(seed=42)
    state = env.current_state()
    assert loaded.action_index(state) == agent.action_index(state)


def test_ppo_runs_evaluation_episode(tmp_path):
    """A trained PPO drives a full evaluation episode without error.

    Confirms the agent integrates with the env's step loop end to end (the path
    the experiment runner will use), producing a finite total reward.
    """
    env, d_bar = _training_env(episode_length=28)
    agent = PPOAgent(mean_daily_demand=d_bar, hyperparams=PPOHyperparams(n_steps=256), seed=0)
    agent.train(env, total_timesteps=2000)

    env.reset(seed=100)
    total_reward = 0.0
    done = False
    while not done:
        action = env.order_units(agent.select_action(env.current_state()))
        _, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
    assert np.isfinite(total_reward)
