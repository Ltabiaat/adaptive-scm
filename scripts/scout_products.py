"""Scout M5 product-store pairs against the thesis selection gates.

The PRD's placeholder pair (FOODS_3_090 / CA_1) does not pass the <10%
zero-sales gate on the real M5 data, usually because the product was not stocked
at that store early in the 2011-2016 window (leading zeros). This script scans
candidate (item_id, store_id) pairs and reports, for each, the three gate
quantities used by ``loader.validate_series`` so a qualifying series can be
chosen with evidence rather than guesswork.

Usage:
    uv run python scripts/scout_products.py
    uv run python scripts/scout_products.py --store CA_3 --top 25
    uv run python scripts/scout_products.py --item FOODS_3_090   # one item, all stores

It only reads ``data/raw``; it writes nothing and trains nothing.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from adaptive_scm.data.loader import (
    CALENDAR_FILE,
    PRICES_FILE,
    SALES_FILE,
    load_m5_series,
)

_ZERO_GATE = 0.10
_YEARS_GATE = 4 * 365


def _gate_metrics(item_id: str, store_id: str, raw_dir: Path) -> dict | None:
    """Return the three gate quantities for one pair, or ``None`` if unloadable.

    Loads the pair through the real loader and computes the same quantities
    ``validate_series`` checks: zero-sales fraction, history span in days, and
    whether every calendar year has at least one event. Reusing the loader keeps
    the verdict identical to what ``preprocess`` will see.

    Args:
        item_id: M5 item id.
        store_id: M5 store id.
        raw_dir: Directory holding the raw M5 CSVs.

    Returns:
        Metrics dict, or ``None`` if the pair could not be loaded.
    """
    try:
        df = load_m5_series(raw_dir, item_id=item_id, store_id=store_id)
    except Exception:
        return None
    zero_frac = float((df["sales"] == 0).mean())
    span = (df["date"].max() - df["date"].min()).days + 1
    events = (
        df.assign(_y=df["date"].dt.year)
        .groupby("_y")["event_name_1"]
        .apply(lambda s: s.notna().sum())
    )
    # Leading-zero diagnostic: zero-fraction after the first non-zero sale.
    first_nz = df["sales"].to_numpy().nonzero()[0]
    if len(first_nz):
        active = df["sales"].to_numpy()[first_nz[0] :]
        active_zero_frac = float((active == 0).mean())
    else:
        active_zero_frac = 1.0
    passes = zero_frac < _ZERO_GATE and span >= _YEARS_GATE and (events >= 1).all()
    return {
        "item_id": item_id,
        "store_id": store_id,
        "zero_frac": zero_frac,
        "active_zero_frac": active_zero_frac,
        "span_days": span,
        "events_ok": bool((events >= 1).all()),
        "passes": bool(passes),
    }


@click.command()
@click.option("--raw-dir", default="data/raw", help="Directory with the M5 CSVs.")
@click.option(
    "--store", default=None, help="Restrict to one store (e.g. CA_3). Default: CA_1..CA_4."
)
@click.option("--item", default=None, help="Scout one item across all stores instead of scanning.")
@click.option("--dept", default="FOODS_3", help="Department prefix to scan (default FOODS_3).")
@click.option("--top", default=20, help="Max candidate items to test when scanning.")
def main(raw_dir: str, store: str | None, item: str | None, dept: str, top: int) -> None:
    """Scan candidate pairs and print which pass the selection gates.

    Ranks scanned items by total sales (busiest first, since high-volume items
    have the fewest zeros) and reports the gate quantities for the top
    candidates. With ``--item`` it instead reports one item across all stores.

    Args:
        raw_dir: Directory with the raw M5 CSVs.
        store: Optional single store filter.
        item: Optional single item (scan its stores).
        dept: Department prefix to scan.
        top: Number of busiest items to test.
    """
    raw_path = Path(raw_dir)
    for f in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        if not (raw_path / f).exists():
            raise click.ClickException(
                f"Missing {raw_path / f}. Put the three M5 CSVs in {raw_dir}."
            )

    sales = pd.read_csv(raw_path / SALES_FILE)
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    if item:
        pairs = [
            (item, s) for s in sorted(sales.loc[sales["item_id"] == item, "store_id"].unique())
        ]
    else:
        sub = sales[sales["item_id"].str.startswith(dept)].copy()
        if store:
            sub = sub[sub["store_id"] == store]
        sub["_total"] = sub[d_cols].sum(axis=1)
        # busiest item-store rows first
        sub = sub.sort_values("_total", ascending=False).head(top)
        pairs = list(zip(sub["item_id"], sub["store_id"]))

    rows = []
    for it, st in pairs:
        m = _gate_metrics(it, st, raw_path)
        if m:
            rows.append(m)

    if not rows:
        raise click.ClickException("No candidate pairs could be loaded.")

    table = pd.DataFrame(rows)
    table = table.sort_values(["passes", "zero_frac"], ascending=[False, True])
    pd.set_option("display.width", 120)
    fmt = table.assign(
        zero_frac=(table["zero_frac"] * 100).round(1).astype(str) + "%",
        active_zero_frac=(table["active_zero_frac"] * 100).round(1).astype(str) + "%",
    )
    click.echo("")
    click.echo(fmt.to_string(index=False))
    click.echo("")
    winners = table[table["passes"]]
    if len(winners):
        best = winners.iloc[0]
        click.echo(
            f"PASS: {len(winners)} pair(s) qualify. Suggested: "
            f"item_id={best['item_id']}, store_id={best['store_id']} "
            f"(zero={best['zero_frac']*100:.1f}%, span={best['span_days']}d)."
        )
        click.echo(
            "Set these under data.product_store in config/default.yaml, then re-run preprocess."
        )
    else:
        click.echo(
            "No pair passed. The 'active_zero_frac' column shows zeros AFTER the first sale: "
            "if those are low, the failures are leading zeros and trimming them would qualify "
            "the series (a documented preprocessing choice)."
        )


if __name__ == "__main__":
    main()
