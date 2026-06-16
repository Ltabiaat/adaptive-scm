"""CLI entrypoint for training a PPO inventory agent.

Loads the processed series and a frozen forecaster, builds a randomized-start
training-episode factory carrying that forecaster's accuracy, trains a PPO agent
on the inventory environment, and saves it to ``results/ppo_{forecaster}.zip``
(PRD Feature 9). The forecaster is loaded read-only and never retrained.

Usage:
    uv run python scripts/train_ppo.py --forecaster=tft
    uv run python scripts/train_ppo.py --forecaster=arima --timesteps 50000
"""

from __future__ import annotations

from adaptive_scm.utils.openmp import allow_duplicate_openmp

# Training PPO on a frozen XGBoost forecaster loads both XGBoost and torch in one
# process; permit OpenMP coexistence on macOS before either is imported (D-4.7).
allow_duplicate_openmp()

from pathlib import Path  # noqa: E402

import click  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from adaptive_scm.policies import PPOAgent, PPOHyperparams  # noqa: E402
from adaptive_scm.simulation import (
    EnvConfig,
    InventoryEnv,
    make_training_episode_factory,
)  # noqa: E402
from adaptive_scm.utils.logging import get_logger  # noqa: E402
from adaptive_scm.utils.seeding import set_global_seed  # noqa: E402

_LOG = get_logger(__name__)
_RESULTS_DIR = Path("results")


def _load_forecaster(name: str):
    """Load a frozen forecaster artifact by name.

    Dispatches on the forecaster name to the right loader and artifact path
    (TFT saves a directory, the others a ``.joblib`` file). Returns the loaded
    forecaster, from which only ``historical_rmse`` is read here.

    Args:
        name: One of ``arima``, ``xgboost``, ``tft``.

    Returns:
        The loaded forecaster instance.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    paths = {
        "arima": _RESULTS_DIR / "forecaster_arima.joblib",
        "xgboost": _RESULTS_DIR / "forecaster_xgboost.joblib",
        "tft": _RESULTS_DIR / "forecaster_tft",
    }
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(
            f"frozen forecaster not found at {path}; train it first with "
            f"scripts/train_forecaster.py --model={name}"
        )
    # Import only the requested forecaster's module (D-4.7: avoid loading both
    # XGBoost and torch into one process on macOS).
    if name == "arima":
        from adaptive_scm.forecasting.arima import ARIMAForecaster as cls
    elif name == "xgboost":
        from adaptive_scm.forecasting.xgboost import XGBoostForecaster as cls
    else:
        from adaptive_scm.forecasting.tft import TFTForecaster as cls
    return cls.load(path)


def _env_config(cfg, training: bool) -> EnvConfig:
    """Build an :class:`EnvConfig` from the merged config.

    Reads the ``simulation`` block; uses the training episode length for PPO
    training and the (shorter) evaluation length otherwise.

    Args:
        cfg: OmegaConf config node.
        training: Whether to use the training episode length.

    Returns:
        A populated :class:`EnvConfig`.
    """
    s = cfg.simulation
    length = s.episode.training_length if training else s.episode.length
    return EnvConfig(
        holding_cost=s.costs.holding_per_unit_per_day,
        stockout_cost=s.costs.stockout_per_unit,
        fixed_order_cost=s.costs.fixed_order,
        purchase_cost=s.costs.purchase_per_unit,
        lead_time_base=s.lead_time.base,
        lead_time_max_additional=s.lead_time.stochastic_max_additional,
        episode_length=length,
        mean_daily_demand=1.0,  # overwritten below from the series
        demand_noise_cv=s.noise.demand_cv,
    )


@click.command()
@click.option(
    "--forecaster",
    type=click.Choice(["arima", "xgboost", "tft"]),
    required=True,
    help="Which frozen forecaster supplies the forecast signal.",
)
@click.option(
    "--config",
    "config_path",
    default="config/default.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML config.",
)
@click.option("--timesteps", default=None, type=int, help="Override total training timesteps.")
@click.option("--seed", default=42, type=int, help="Global random seed.")
def main(forecaster: str, config_path: Path, timesteps: int | None, seed: int) -> None:
    """Train and save a PPO agent for the chosen frozen forecaster.

    Seeds RNGs, loads the processed series and the frozen forecaster, builds a
    randomized-start training-episode factory, trains PPO on the inventory
    environment, and writes the agent to ``results/ppo_{forecaster}.zip``.

    Args:
        forecaster: Frozen forecaster name.
        config_path: Path to the YAML config.
        timesteps: Optional override of ``policies.ppo.total_timesteps``.
        seed: Global random seed.

    Raises:
        FileNotFoundError: If processed data or the forecaster artifact is missing.
    """
    set_global_seed(seed)
    cfg = OmegaConf.load(config_path)

    item_id = cfg.data.product_store.item_id
    store_id = cfg.data.product_store.store_id
    processed = Path(cfg.data.processed_dir) / f"{item_id}_{store_id}.parquet"
    if not processed.exists():
        raise FileNotFoundError(f"processed data not found at {processed}; run preprocess first")
    df = pd.read_parquet(processed)
    train_df = df[df["split"] == "train"]
    sales = train_df["sales"].to_numpy(dtype=float)
    dow = pd.to_datetime(train_df["date"]).dt.dayofweek.to_numpy()

    frozen = _load_forecaster(forecaster)
    rmse = float(frozen.historical_rmse)

    env_cfg = _env_config(cfg, training=True)
    d_bar = float(sales.mean())
    env_cfg = EnvConfig(**{**env_cfg.__dict__, "mean_daily_demand": d_bar})

    factory = make_training_episode_factory(
        sales=sales,
        day_of_week=dow,
        historical_rmse=rmse,
        episode_length=env_cfg.episode_length,
    )
    # A seed episode for construction; the factory replaces it each reset.
    seed_episode = factory(__import__("numpy").random.default_rng(seed))
    env = InventoryEnv(env_cfg, seed_episode, seed=seed, episode_factory=factory)

    p = cfg.policies.ppo
    hp = PPOHyperparams(
        clip_range=p.clip_range,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        learning_rate=p.learning_rate,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        n_steps=p.n_steps,
        net_arch=tuple(p.policy_net),
    )
    total = timesteps if timesteps is not None else p.total_timesteps

    agent = PPOAgent(mean_daily_demand=d_bar, hyperparams=hp, seed=seed)
    agent.train(env, total_timesteps=total)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"ppo_{forecaster}.zip"
    agent.save(out_path)
    _LOG.info("ppo_training_done", forecaster=forecaster, timesteps=total, output=str(out_path))


if __name__ == "__main__":
    main()
