"""Hypothesis tests for the thesis (PRD Feature 12).

Turns the per-cell simulation Parquets into statistical evidence for the three
thesis hypotheses:

* **H1** -- a learned policy (PPO) beats the classical policies (EOQ,
  order-up-to), holding the forecaster constant. Paired t-tests on per-
  replication total cost; replications are paired because cell ``r`` of every
  policy shares replication seed ``base + r`` and therefore the same demand and
  lead-time draws.
* **H2** -- the fully integrated TFT+PPO pipeline beats the standalone
  classical-policy baselines.
* **H3** -- forecast accuracy does not determine decision quality: Spearman
  correlation between forecast RMSE and inventory cost across combinations.

Pure functions here; ``scripts/run_analysis.py`` wires them to disk and renders
``results/analysis/hypothesis_tests.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from adaptive_scm.utils.logging import get_logger

_LOG = get_logger(__name__)

_FORECASTERS = ("arima", "xgboost", "tft")
_CLASSICAL_POLICIES = ("eoq", "order_up_to")
_CONDITIONS = ("baseline", "demand_spike", "lead_time_disruption")
_ALPHA = 0.05


def per_replication_costs(cell: pd.DataFrame) -> np.ndarray:
    """Reconstruct per-replication total cost from a cell's daily rows.

    Sums each replication's daily ``holding + stockout + order`` costs, returned
    ordered by replication index. Because replication ``r`` uses the same seed
    across policies, the index ordering aligns pairs across cells.

    Args:
        cell: A loaded experiment Parquet (daily + summary rows).

    Returns:
        Array of per-replication total costs, length ``n_replications``.
    """
    daily = cell[cell["record_type"] == "daily"]
    grouped = daily.groupby("replication")[["holding_cost", "stockout_cost", "order_cost"]].sum()
    return (grouped["holding_cost"] + grouped["stockout_cost"] + grouped["order_cost"]).to_numpy()


def _load_costs(sim_dir: Path, forecaster: str, policy: str, condition: str) -> np.ndarray | None:
    """Load per-replication costs for one cell, or ``None`` if missing.

    Args:
        sim_dir: Directory of per-cell Parquets.
        forecaster: Forecaster label.
        policy: Policy label.
        condition: Condition label.

    Returns:
        Per-replication cost array, or ``None`` if the file is absent.
    """
    path = sim_dir / f"{forecaster}_{policy}_{condition}.parquet"
    if not path.exists():
        return None
    return per_replication_costs(pd.read_parquet(path))


def paired_comparison(better: np.ndarray, worse: np.ndarray, label: str) -> dict:
    """Paired t-test of two policies' per-replication costs (lower is better).

    Aligns the two cost arrays by replication index (truncating to the shorter
    if needed), runs ``ttest_rel`` on the differences, and computes the paired
    effect size (Cohen's d_z = mean(diff) / sd(diff)). The ``label`` policy is
    "better" when its mean cost is lower.

    Args:
        better: Per-replication costs of the policy hypothesized to win.
        worse: Per-replication costs of the comparison policy.
        label: Short label for the comparison (e.g. ``"ppo_vs_eoq"``).

    Returns:
        Dict with means, mean difference, t-stat, p-value, Cohen's d, n, and a
        boolean ``favors_better`` (mean cost lower) and ``significant``.
    """
    n = min(len(better), len(worse))
    a, b = better[:n], worse[:n]
    diff = a - b  # negative => 'better' policy has lower cost
    t_stat, p_value = ttest_rel(a, b)
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    cohens_d = float(diff.mean() / sd) if sd > 1e-12 else 0.0
    return {
        "comparison": label,
        "mean_cost_better": float(a.mean()),
        "mean_cost_worse": float(b.mean()),
        "mean_difference": float(diff.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n": n,
        "favors_better": bool(a.mean() < b.mean()),
        "significant": bool(p_value < _ALPHA),
    }


def test_h1(sim_dir: Path, conditions: tuple[str, ...] = _CONDITIONS) -> list[dict]:
    """H1: PPO vs EOQ and PPO vs order-up-to, per forecaster and condition.

    For each forecaster and condition where all needed cells exist, runs a
    paired comparison of PPO against each classical policy. The PPO-vs-order-up-to
    comparison is the thesis's load-bearing test (both policies receive identical
    forecast information, so a difference isolates the value of learning).

    Args:
        sim_dir: Directory of per-cell Parquets.
        conditions: Conditions to test.

    Returns:
        List of comparison dicts, each tagged with ``forecaster`` and ``condition``.
    """
    rows: list[dict] = []
    for forecaster in _FORECASTERS:
        for condition in conditions:
            ppo = _load_costs(sim_dir, forecaster, "ppo", condition)
            if ppo is None:
                continue
            for classical in _CLASSICAL_POLICIES:
                other = _load_costs(sim_dir, forecaster, classical, condition)
                if other is None:
                    continue
                row = paired_comparison(ppo, other, f"ppo_vs_{classical}")
                row.update(forecaster=forecaster, condition=condition)
                rows.append(row)
    return rows


def test_h2(sim_dir: Path, condition: str = "baseline") -> list[dict]:
    """H2: integrated TFT+PPO vs the six classical-policy baselines.

    Compares TFT+PPO against each ``(forecaster, classical_policy)`` combination
    -- the six standalone baselines that use a non-learned inventory policy
    (D-12.1). Run at one condition (baseline by default) to keep the integrated-
    vs-standalone contrast clean.

    Args:
        sim_dir: Directory of per-cell Parquets.
        condition: Condition to test at.

    Returns:
        List of comparison dicts vs each baseline; empty if TFT+PPO is missing.
    """
    integrated = _load_costs(sim_dir, "tft", "ppo", condition)
    if integrated is None:
        return []
    rows: list[dict] = []
    for forecaster in _FORECASTERS:
        for policy in _CLASSICAL_POLICIES:
            baseline = _load_costs(sim_dir, forecaster, policy, condition)
            if baseline is None:
                continue
            row = paired_comparison(integrated, baseline, f"tft_ppo_vs_{forecaster}_{policy}")
            row.update(condition=condition)
            rows.append(row)
    return rows


def test_h3(suite: pd.DataFrame) -> dict:
    """H3: Spearman correlation between forecast RMSE and inventory cost.

    Correlates ``forecast_rmse`` against ``total_cost_mean`` across all
    combinations in the suite summary. A weak or non-significant correlation
    supports H3 -- accuracy does not directly translate into decision quality.

    Args:
        suite: The suite summary table (one row per combination, with
            ``forecast_rmse`` and ``total_cost_mean``).

    Returns:
        Dict with ``spearman_rho``, ``p_value``, ``n``, and ``supports_h3``.
    """
    if "forecast_rmse" not in suite or len(suite) < 3:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n": len(suite),
            "supports_h3": False,
        }
    rmse, cost = suite["forecast_rmse"], suite["total_cost_mean"]
    if rmse.nunique() < 2 or cost.nunique() < 2:
        return {
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
            "n": len(suite),
            "supports_h3": False,
        }
    rho, p = spearmanr(rmse, cost)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": int(len(suite)),
        "supports_h3": bool(p >= _ALPHA or abs(rho) < 0.3),
    }


def render_hypothesis_markdown(h1: list[dict], h2: list[dict], h3: dict) -> str:
    """Render the three hypothesis tests as a Markdown report.

    Lays out H1 and H2 as tables of comparisons (means, mean difference,
    p-value, Cohen's d, verdict) and H3 as a one-line correlation result, each
    with a short interpretation. Used to write ``hypothesis_tests.md``.

    Args:
        h1: H1 comparison rows.
        h2: H2 comparison rows.
        h3: H3 correlation result.

    Returns:
        The report as a Markdown string.
    """
    lines = ["# Hypothesis Tests", ""]
    lines.append(
        f"Paired t-tests on per-replication total cost (alpha = {_ALPHA}); "
        f"lower cost is better. Effect size is Cohen's d_z. P-values are raw "
        f"(uncorrected for multiple comparisons -- interpret with care)."
    )
    lines.append("")

    lines.append("## H1 -- learned policy (PPO) vs classical policies")
    lines.append("")
    lines.append(_comparison_table(h1, group_cols=("forecaster", "condition")))
    lines.append("")
    lines.append(_h1_interpretation(h1))
    lines.append("")

    lines.append("## H2 -- integrated TFT+PPO vs standalone baselines")
    lines.append("")
    lines.append(_comparison_table(h2, group_cols=("condition",)))
    lines.append("")
    lines.append(_h2_interpretation(h2))
    lines.append("")

    lines.append("## H3 -- forecast accuracy vs decision quality")
    lines.append("")
    if np.isnan(h3["spearman_rho"]):
        lines.append("Not enough varied data to compute the RMSE-cost correlation.")
    else:
        verdict = "supports H3" if h3["supports_h3"] else "does not support H3"
        lines.append(
            f"Spearman rho = {h3['spearman_rho']:.3f} (p = {h3['p_value']:.3f}, "
            f"n = {h3['n']} combinations). This {verdict}: a weak or "
            f"non-significant correlation means forecast accuracy does not "
            f"directly determine inventory cost."
        )
    lines.append("")
    return "\n".join(lines)


def _comparison_table(rows: list[dict], group_cols: tuple[str, ...]) -> str:
    """Format comparison rows as a Markdown table.

    Args:
        rows: Comparison dicts from :func:`paired_comparison` (with group tags).
        group_cols: Leading label columns to show (e.g. forecaster/condition).

    Returns:
        A Markdown table, or a placeholder if there are no rows.
    """
    if not rows:
        return "_No results available (missing result files)._"
    df = pd.DataFrame(rows)
    cols = list(group_cols) + [
        "comparison",
        "mean_cost_better",
        "mean_cost_worse",
        "mean_difference",
        "p_value",
        "cohens_d",
        "significant",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].round(3).to_markdown(index=False)


def _h1_interpretation(rows: list[dict]) -> str:
    """Summarize H1 in prose: how often PPO wins, and significantly.

    Args:
        rows: H1 comparison rows.

    Returns:
        A one-paragraph interpretation.
    """
    if not rows:
        return "_No H1 results available._"
    wins = sum(r["favors_better"] for r in rows)
    sig_wins = sum(r["favors_better"] and r["significant"] for r in rows)
    oup = [r for r in rows if r["comparison"] == "ppo_vs_order_up_to"]
    oup_sig = sum(r["favors_better"] and r["significant"] for r in oup)
    return (
        f"PPO had lower mean cost in {wins} of {len(rows)} comparisons, "
        f"{sig_wins} of them significant at {_ALPHA}. Against order-up-to "
        f"specifically -- the comparison that isolates learning from information "
        f"-- PPO won significantly in {oup_sig} of {len(oup)} cases."
    )


def _h2_interpretation(rows: list[dict]) -> str:
    """Summarize H2 in prose: how often TFT+PPO beats the baselines.

    Args:
        rows: H2 comparison rows.

    Returns:
        A one-paragraph interpretation.
    """
    if not rows:
        return "_No H2 results available._"
    wins = sum(r["favors_better"] for r in rows)
    sig = sum(r["favors_better"] and r["significant"] for r in rows)
    return (
        f"TFT+PPO had lower mean cost than {wins} of {len(rows)} standalone "
        f"baselines, {sig} significantly at {_ALPHA}."
    )
