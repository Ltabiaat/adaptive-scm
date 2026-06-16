"""CLI entrypoint for one experiment: a (forecaster, policy, condition) cell.

Loads the frozen forecaster and processed series, builds the evaluation episode,
constructs the requested policy, wraps the environment for the disruption
condition, runs N replications, and saves the per-day records plus an aggregate
summary to ``results/simulations/{forecaster}_{policy}_{condition}.parquet``
(PRD Feature 10).

Usage:
    uv run python scripts/run_experiment.py --forecaster=arima --policy=eoq \\
        --condition=baseline --replications=30
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from adaptive_scm.simulation import (
    DemandSpikeWrapper,
    EnvConfig,
    InventoryEnv,
    LeadTimeDisruptionWrapper,
    build_eval_episode,
    result_to_dataframe,
    run_replications,
)
from adaptive_scm.utils.logging import get_logger
from adaptive_scm.utils.seeding import set_global_seed

_LOG = get_logger(__name__)
_RESULTS_DIR = Path("results")
_SIM_DIR = _RESULTS_DIR / "simulations"


def load_frozen_forecaster(name: str):
    """Load a frozen forecaster by name (read-only).

    Args:
        name: ``arima`` / ``xgboost`` / ``tft``.

    Returns:
        The loaded forecaster (only ``historical_rmse`` is used downstream).

    Raises:
        FileNotFoundError: If the artifact is missing.
    """
    from adaptive_scm.forecasting import ARIMAForecaster, TFTForecaster, XGBoostForecaster

    loaders = {
        "arima": (ARIMAForecaster, _RESULTS_DIR / "forecaster_arima.joblib"),
        "xgboost": (XGBoostForecaster, _RESULTS_DIR / "forecaster_xgboost.joblib"),
        "tft": (TFTForecaster, _RESULTS_DIR / "forecaster_tft"),
    }
    cls, path = loaders[name]
    if not path.exists():
        raise FileNotFoundError(f"forecaster {name!r} not found at {path}; train it first")
    return cls.load(path)


def build_env_config(cfg, mean_daily_demand: float) -> EnvConfig:
    """Build the evaluation :class:`EnvConfig` from the merged config.

    Args:
        cfg: OmegaConf config node.
        mean_daily_demand: ``d_bar`` for the action grid.

    Returns:
        A populated :class:`EnvConfig` (evaluation episode length).
    """
    s = cfg.simulation
    return EnvConfig(
        holding_cost=s.costs.holding_per_unit_per_day,
        stockout_cost=s.costs.stockout_per_unit,
        fixed_order_cost=s.costs.fixed_order,
        purchase_cost=s.costs.purchase_per_unit,
        lead_time_base=s.lead_time.base,
        lead_time_max_additional=s.lead_time.stochastic_max_additional,
        episode_length=s.episode.length,
        mean_daily_demand=mean_daily_demand,
        demand_noise_cv=s.noise.demand_cv,
    )


def build_policy(name: str, cfg, env_cfg: EnvConfig, forecaster_name: str):
    """Construct the requested policy from config.

    EOQ and OrderUpTo are built from cost/lead-time/service config; PPO is
    loaded from its trained artifact (forecaster-specific) with the env ``d_bar``.

    Args:
        name: ``eoq`` / ``order_up_to`` / ``ppo``.
        cfg: OmegaConf config node.
        env_cfg: The evaluation env config (for costs / lead time / ``d_bar``).
        forecaster_name: Which PPO artifact to load (``ppo_{forecaster}.zip``).

    Returns:
        A constructed policy.

    Raises:
        FileNotFoundError: If the PPO artifact is missing.
    """
    from adaptive_scm.policies import EOQPolicy, OrderUpToPolicy

    if name == "eoq":
        return EOQPolicy(
            holding_per_unit_per_day=env_cfg.holding_cost,
            fixed_order_cost=env_cfg.fixed_order_cost,
            lead_time=env_cfg.lead_time_base,
            service_level=cfg.policies.eoq.service_level,
        )
    if name == "order_up_to":
        return OrderUpToPolicy(
            lead_time=env_cfg.lead_time_base,
            review_period=cfg.policies.order_up_to.review_period,
            service_level=cfg.policies.order_up_to.service_level,
        )
    from adaptive_scm.policies import PPOAgent

    path = _RESULTS_DIR / f"ppo_{forecaster_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"PPO agent not found at {path}; train it with train_ppo.py")
    return PPOAgent.load(path, mean_daily_demand=env_cfg.mean_daily_demand)


def wrap_condition(env, condition: str, cfg):
    """Wrap the env for a disruption condition and return its analysis window.

    Args:
        env: The base inventory environment.
        condition: ``baseline`` / ``demand_spike`` / ``lead_time_disruption``.
        cfg: OmegaConf config node (disruption window + multipliers).

    Returns:
        Tuple ``(wrapped_env, disruption_window)`` where the window is ``None``
        for baseline.

    Raises:
        ValueError: If ``condition`` is unknown.
    """
    e = cfg.experiments
    start = e.disruption_window.start_day
    duration = e.disruption_window.duration_days
    window = (start, start + duration)

    if condition == "baseline":
        return env, None
    if condition == "demand_spike":
        return (
            DemandSpikeWrapper(env, e.demand_spike_multiplier, start, duration),
            window,
        )
    if condition == "lead_time_disruption":
        return (
            LeadTimeDisruptionWrapper(env, e.lead_time_disruption_multiplier, start, duration),
            window,
        )
    raise ValueError(f"unknown condition: {condition!r}")


@click.command()
@click.option("--forecaster", type=click.Choice(["arima", "xgboost", "tft"]), required=True)
@click.option("--policy", type=click.Choice(["eoq", "order_up_to", "ppo"]), required=True)
@click.option(
    "--condition",
    type=click.Choice(["baseline", "demand_spike", "lead_time_disruption"]),
    required=True,
)
@click.option("--replications", default=30, type=int)
@click.option(
    "--config",
    "config_path",
    default="config/default.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--seed", default=42, type=int)
def main(
    forecaster: str,
    policy: str,
    condition: str,
    replications: int,
    config_path: Path,
    seed: int,
) -> None:
    """Run one experiment cell and persist its results.

    Args:
        forecaster: Frozen forecaster supplying the forecast signal.
        policy: Replenishment policy to evaluate.
        condition: Disruption condition.
        replications: Number of replications.
        config_path: Path to the YAML config.
        seed: Base seed; replication ``r`` uses ``seed + r``.

    Raises:
        FileNotFoundError: If processed data or a required artifact is missing.
    """
    set_global_seed(seed)
    cfg = OmegaConf.load(config_path)

    item_id = cfg.data.product_store.item_id
    store_id = cfg.data.product_store.store_id
    processed = Path(cfg.data.processed_dir) / f"{item_id}_{store_id}.parquet"
    if not processed.exists():
        raise FileNotFoundError(f"processed data not found at {processed}")
    df = pd.read_parquet(processed)
    dow_all = pd.to_datetime(df["date"]).dt.dayofweek.to_numpy()
    d_bar = float(df[df["split"] == "train"]["sales"].mean())

    frozen = load_frozen_forecaster(forecaster)
    rmse = float(frozen.historical_rmse)

    env_cfg = build_env_config(cfg, d_bar)
    horizon = env_cfg.episode_length

    # The forecast the policy sees is the forecaster's actual prediction for the
    # test window; the ground truth is the realized test-split sales (D-9.7).
    prediction = np.asarray(frozen.predict(horizon).point_forecast, dtype=float)
    test_df = df[df["split"] == "test"].head(horizon)
    realized = test_df["sales"].to_numpy(dtype=float)
    test_dow = dow_all[df["split"].to_numpy() == "test"][:horizon]

    episode = build_eval_episode(prediction, realized, test_dow, rmse, horizon)
    pol = build_policy(policy, cfg, env_cfg, forecaster)

    base_env = InventoryEnv(env_cfg, episode, seed=seed)
    env, window = wrap_condition(base_env, condition, cfg)
    seeds = [seed + r for r in range(replications)]

    result = run_replications(env, pol, replications, seeds=seeds, disruption_window=window)
    result.summary["forecast_rmse"] = rmse

    _SIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _SIM_DIR / f"{forecaster}_{policy}_{condition}.parquet"
    out_df = result_to_dataframe(result)
    out_df.loc[out_df["record_type"] == "summary", "forecast_rmse"] = rmse
    out_df.to_parquet(out_path, index=False)

    _LOG.info(
        "experiment_saved",
        forecaster=forecaster,
        policy=policy,
        condition=condition,
        replications=replications,
        total_cost_mean=result.summary["total_cost_mean"],
        fill_rate_mean=result.summary["fill_rate_mean"],
        output=str(out_path),
    )


if __name__ == "__main__":
    main()
