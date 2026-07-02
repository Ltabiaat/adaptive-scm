"""Diagnostic: daily fill-rate trajectory for one cell, disruption vs baseline.

Reads the saved per-(replication, day) rows from two simulation parquets -- a
disruption cell and its baseline -- averages daily fill rate across replications,
and prints/plots the trajectories with the disruption window shaded. This makes
visible *why* a recovery-time metric is what it is: a lead-time disruption
typically dips during its window then rebounds immediately (no recovery tail),
whereas a demand spike depletes stock and leaves a multi-day tail.

Daily fill rate = 1 - lost_sales / demand (days with zero demand count as 1.0).

Usage:
    uv run python scripts/diagnose_recovery.py --forecaster arima --policy ppo \\
        --condition lead_time_disruption
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

_SIM_DIR = Path("results/simulations")


def _daily_fill(path: Path) -> np.ndarray:
    """Return mean daily fill rate across replications for one cell parquet.

    Groups the per-(replication, day) rows by day and averages
    ``1 - lost_sales/demand`` across replications, treating zero-demand days as
    fully served. Used to compare a disruption trajectory against its baseline.

    Args:
        path: Path to a per-cell simulation parquet (daily rows).

    Returns:
        Array of length = episode horizon, mean fill rate per day.
    """
    df = pd.read_parquet(path)
    df = df[df["day"].notna()].copy()
    demand = df["demand"].to_numpy(dtype=float)
    lost = df["lost_sales"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(demand > 0, 1.0 - lost / demand, 1.0)
    df["fill"] = fill
    return df.groupby("day")["fill"].mean().to_numpy()


@click.command()
@click.option("--forecaster", default="arima")
@click.option("--policy", default="ppo")
@click.option("--condition", default="lead_time_disruption")
@click.option("--window-start", default=7, help="Disruption window start day (config).")
@click.option("--window-len", default=14, help="Disruption window length (config).")
@click.option("--plot/--no-plot", default=True, help="Also save a PNG trajectory plot.")
def main(
    forecaster: str, policy: str, condition: str, window_start: int, window_len: int, plot: bool
) -> None:
    """Print (and optionally plot) the daily fill-rate trajectory vs baseline.

    Loads the chosen disruption cell and its baseline counterpart, prints a
    day-by-day table of both with the disruption window flagged, and reports the
    post-window day on which the disrupted series re-enters one point of baseline.

    Args:
        forecaster: Forecaster name.
        policy: Policy name.
        condition: Disruption condition to inspect.
        window_start: First day of the disruption window.
        window_len: Length of the disruption window in days.
        plot: Whether to save a PNG.
    """
    disr = _SIM_DIR / f"{forecaster}_{policy}_{condition}.parquet"
    base = _SIM_DIR / f"{forecaster}_{policy}_baseline.parquet"
    for p in (disr, base):
        if not p.exists():
            raise click.ClickException(f"missing {p}; run the suite first")

    fill_d = _daily_fill(disr)
    fill_b = _daily_fill(base)
    n = min(len(fill_d), len(fill_b))
    win_end = window_start + window_len

    click.echo(f"\n{forecaster} / {policy} / {condition}  vs  baseline")
    click.echo(f"disruption window: days {window_start}-{win_end - 1} (shaded)\n")
    click.echo(f"{'day':>4} {'baseline':>9} {'disrupted':>10} {'gap':>7}  window")
    for d in range(n):
        in_win = window_start <= d < win_end
        gap = fill_b[d] - fill_d[d]
        flag = "  <== disrupted" if in_win else ""
        click.echo(f"{d:>4} {fill_b[d]:>9.3f} {fill_d[d]:>10.3f} {gap:>7.3f}{flag}")

    # First post-window day back within 1 point of baseline.
    recovered = None
    for d in range(win_end, n):
        if fill_b[d] - fill_d[d] <= 0.01:
            recovered = d
            break
    if recovered is None:
        click.echo("\nDid not return within 1 point of baseline before horizon end.")
    elif recovered == win_end:
        click.echo(f"\nBack within 1 point of baseline immediately at window end (day {win_end}).")
    else:
        click.echo(
            f"\nReturned within 1 point of baseline on day {recovered} "
            f"({recovered - win_end} day(s) after the window)."
        )

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        days = np.arange(n)
        ax.plot(days, fill_b[:n], label="baseline", color="#2c7fb8", lw=2)
        ax.plot(days, fill_d[:n], label=condition, color="#d95f0e", lw=2)
        ax.axvspan(window_start, win_end - 1, color="grey", alpha=0.15, label="disruption window")
        ax.set_xlabel("day")
        ax.set_ylabel("mean daily fill rate")
        ax.set_title(f"{forecaster} / {policy}: {condition} vs baseline")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        out = Path("results/analysis") / f"recovery_{forecaster}_{policy}_{condition}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        click.echo(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
