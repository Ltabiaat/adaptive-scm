"""Inventory simulation: Gymnasium environment, disruptions, and the runner.

Public surface:
    - ``InventoryEnv``: single-product, single-echelon daily inventory env.
    - ``EnvConfig``: typed configuration for costs, lead time, and episode shape.
    - ``EpisodeData``: the per-episode forecast / demand / calendar inputs.
"""

from adaptive_scm.simulation.disruptions import (
    DemandSpikeWrapper,
    LeadTimeDisruptionWrapper,
)
from adaptive_scm.simulation.environment import EnvConfig, EpisodeData, InventoryEnv
from adaptive_scm.simulation.episodes import (
    build_eval_episode,
    build_forecast_arrays,
    make_training_episode_factory,
)
from adaptive_scm.simulation.runner import (
    ExperimentResult,
    result_to_dataframe,
    run_replications,
)

__all__ = [
    "InventoryEnv",
    "EnvConfig",
    "EpisodeData",
    "DemandSpikeWrapper",
    "LeadTimeDisruptionWrapper",
    "build_forecast_arrays",
    "build_eval_episode",
    "make_training_episode_factory",
    "run_replications",
    "result_to_dataframe",
    "ExperimentResult",
]
