"""Multi-replication experiment runner.

Drives a single ``(policy, environment, condition)`` through N replications,
each with a distinct seed, collecting per-day records and reducing them to
per-replication and aggregate metrics. The runner is policy-agnostic: it calls
``policy.select_action`` and maps the result to a discrete action via
``env.order_units``, so EOQ, OrderUpTo, and PPO are driven identically (D-7.2 /
D-9.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adaptive_scm.evaluation.metrics import aggregate_metrics, compute_episode_metrics
from adaptive_scm.policies.base import Policy
from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Outputs of one experiment (all replications of one combination).

    Attributes:
        daily: One row per ``(replication, day)`` with demand, order, inventory,
            costs, and reward.
        per_replication: One metric dict per replication.
        summary: Aggregate metrics across replications (means + total-cost std).
    """

    daily: pd.DataFrame
    per_replication: list[dict]
    summary: dict[str, float]


def run_replications(
    env,
    policy: Policy,
    n_replications: int,
    seeds: list[int] | None = None,
) -> ExperimentResult:
    """Run ``n_replications`` episodes of ``policy`` in ``env`` and collect metrics.

    Each replication resets the policy and the environment with a distinct seed,
    steps through one full episode collecting the env's per-day ``info`` plus
    reward, then computes that replication's cost/service metrics. Per-replication
    metrics are aggregated into the summary. The same call works for any policy
    and any (already-wrapped) environment; the disruption condition is handled
    entirely by the caller's choice of ``env``. Resilience metrics are computed
    later at the suite level (cross-condition), not here.

    Args:
        env: A Gymnasium inventory environment, optionally disruption-wrapped.
        policy: The policy to evaluate.
        n_replications: Number of replications to run.
        seeds: Optional explicit per-replication seeds; defaults to ``0..N-1``.

    Returns:
        An :class:`ExperimentResult`.

    Raises:
        ValueError: If ``n_replications`` is not positive or ``seeds`` length
            does not match.
    """
    if n_replications < 1:
        raise ValueError(f"n_replications must be >= 1, got {n_replications}")
    if seeds is None:
        seeds = list(range(n_replications))
    if len(seeds) != n_replications:
        raise ValueError(f"need {n_replications} seeds, got {len(seeds)}")

    daily_frames: list[pd.DataFrame] = []
    per_rep: list[dict] = []

    for rep, seed in enumerate(seeds):
        records = _run_one_episode(env, policy, seed)
        per_rep.append(compute_episode_metrics(records))
        frame = pd.DataFrame(records)
        frame.insert(0, "day", range(len(frame)))
        frame.insert(0, "replication", rep)
        daily_frames.append(frame)

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = aggregate_metrics(per_rep)
    _LOG.info(
        "experiment_complete",
        n_replications=n_replications,
        total_cost_mean=summary["total_cost_mean"],
        fill_rate_mean=summary["fill_rate_mean"],
    )
    return ExperimentResult(daily=daily, per_replication=per_rep, summary=summary)


def _run_one_episode(env, policy: Policy, seed: int) -> list[dict]:
    """Run a single episode and return its per-day records.

    Resets the policy and env (the env regenerates stochastic demand from the
    seed), then steps until termination, each day mapping the policy's order
    quantity to a discrete action and recording the env ``info`` plus reward.

    Args:
        env: The (possibly wrapped) environment.
        policy: The policy to drive.
        seed: Replication seed.

    Returns:
        List of per-day record dicts.
    """
    policy.reset()
    env.reset(seed=seed)
    base_env = env.unwrapped

    records: list[dict] = []
    done = False
    while not done:
        state = base_env.current_state()
        action = base_env.order_units(policy.select_action(state))
        _, reward, terminated, truncated, info = env.step(action)
        info = dict(info)
        info["reward"] = float(reward)
        info["sales"] = info["demand"] - info["lost_sales"]
        records.append(info)
        done = terminated or truncated
    return records


def result_to_dataframe(result: ExperimentResult) -> pd.DataFrame:
    """Flatten an :class:`ExperimentResult` into one persistable DataFrame.

    Produces the PRD Feature 10 layout: every daily row (``record_type='daily'``)
    plus a single aggregate ``record_type='summary'`` row carrying the summary
    metrics. Daily and summary columns coexist with NaNs where not applicable,
    so the whole experiment fits one Parquet file.

    Args:
        result: The experiment result to flatten.

    Returns:
        A combined DataFrame ready to write to Parquet.
    """
    daily = result.daily.copy()
    daily["record_type"] = "daily"
    summary = pd.DataFrame([{**result.summary, "record_type": "summary"}])
    return pd.concat([daily, summary], ignore_index=True)
