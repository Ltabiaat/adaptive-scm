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


# --------------------------------------------------------------------------- #
# Full-suite orchestration (Feature 11)
# --------------------------------------------------------------------------- #


def test_full_run(tmp_path, monkeypatch):
    """PRD Feature 11 gate: a reduced suite runs end to end and aggregates.

    Builds the minimal artifacts a suite needs (a processed series plus one
    frozen forecaster and one PPO agent), then drives the runner across a
    reduced set of combinations directly (not via subprocess) and renders the
    summary report -- exercising the same code path ``run_full_suite.py`` uses.
    """
    import numpy as np

    from adaptive_scm.evaluation import collect_summary_rows, render_summary_markdown
    from adaptive_scm.policies import EOQPolicy, PPOAgent, PPOHyperparams
    from adaptive_scm.simulation import (
        EnvConfig,
        InventoryEnv,
        build_eval_episode,
        make_training_episode_factory,
        result_to_dataframe,
        run_replications,
    )

    # Synthetic processed series.
    rng = np.random.default_rng(0)
    length = 400
    days = np.arange(length)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 1, length), 1, None)
    dow = days % 7
    d_bar = float(sales.mean())
    rmse = 2.0

    # One trained PPO agent (tiny budget) standing in for the suite's agents.
    train_cfg = EnvConfig(episode_length=120, mean_daily_demand=d_bar)
    factory = make_training_episode_factory(sales, dow, rmse, 120)
    train_env = InventoryEnv(train_cfg, factory(rng), seed=0, episode_factory=factory)
    agent = PPOAgent(d_bar, PPOHyperparams(n_steps=128, n_epochs=2), seed=0)
    agent.train(train_env, total_timesteps=256)

    eval_cfg = EnvConfig(episode_length=28, mean_daily_demand=d_bar, demand_noise_cv=0.25)
    episode = build_eval_episode(sales[-28:], sales[-28:], dow[-28:], rmse, 28)

    # Reduced suite: 1 forecaster x {eoq, ppo} x {baseline}.
    summaries = []
    sim_dir = tmp_path / "simulations"
    sim_dir.mkdir()
    policies = {"eoq": EOQPolicy(0.05, 10.0, 3), "ppo": agent}
    for policy_name, policy in policies.items():
        env = InventoryEnv(eval_cfg, episode, seed=42)
        result = run_replications(env, policy, n_replications=3, disruption_window=None)
        result.summary["forecast_rmse"] = rmse
        out_path = sim_dir / f"arima_{policy_name}_baseline.parquet"
        result_to_dataframe(result).to_parquet(out_path, index=False)
        assert out_path.exists()
        row = {
            **result.summary,
            "forecaster": "arima",
            "policy": policy_name,
            "condition": "baseline",
        }
        summaries.append(row)

    suite = collect_summary_rows(summaries)
    assert len(suite) == 2
    report = render_summary_markdown(suite)
    (tmp_path / "summary.md").write_text(report)
    assert "# Experimental Suite Summary" in report
    assert "Total cost" in report
