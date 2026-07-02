"""Exploratory data analysis for the configured M5 product-store series.

Reconstructs the daily demand series for a single M5 product-store pair from the
raw competition files (sales, calendar, prices), then produces a set of figures
and summary statistics. Each figure is chosen to justify a specific downstream
modeling decision (lag features, seasonal ARIMA order, price/event features,
choice of RMSE, etc.), so the EDA output maps one-to-one onto the methodology.

Run before the forecasting pipeline: the analysis here motivates the pipeline's
feature and model choices rather than depending on it. Reads raw CSVs directly
and does not import the adaptive_scm package.

Usage:
    python run_eda.py --raw-dir data/raw --item FOODS_3_694 --store CA_1 \
        --out results/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid", context="talk")
PROMO_THRESHOLD = 0.95  # relative price below which a day is flagged promotional


def load_series(raw_dir: Path, item_id: str, store_id: str) -> pd.DataFrame:
    """Reconstruct one product-store daily series from raw M5 files.

    Melts the wide sales row for the target id to long form, merges calendar
    (dates, weekday, events, SNAP) and weekly sell prices, and derives a
    relative price index and promotional flag. Called once at the start of the
    EDA run; every figure function consumes the frame it returns.

    Args:
        raw_dir: Directory holding sales_train_evaluation.csv, calendar.csv,
            and sell_prices.csv.
        item_id: M5 item identifier, e.g. "FOODS_3_694".
        store_id: M5 store identifier, e.g. "CA_1".

    Returns:
        A daily DataFrame sorted by date with columns: date, sales, sell_price,
        rel_price, is_promo, has_event, event_type, dow (0=Mon), day_name,
        month, snap.
    """
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")

    row = sales[(sales["item_id"] == item_id) & (sales["store_id"] == store_id)]
    if row.empty:
        raise ValueError(f"No series found for {item_id} / {store_id}")

    d_cols = [c for c in sales.columns if c.startswith("d_")]
    long = row.melt(id_vars=["item_id", "store_id"], value_vars=d_cols,
                    var_name="d", value_name="sales")

    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_type_1", "snap_CA"]
    long = long.merge(calendar[cal_cols], on="d", how="left")

    store_prices = prices[(prices["item_id"] == item_id) & (prices["store_id"] == store_id)]
    long = long.merge(store_prices[["wm_yr_wk", "sell_price"]], on="wm_yr_wk", how="left")

    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values("date").reset_index(drop=True)
    long["sales"] = long["sales"].astype(float)

    mean_price = long["sell_price"].mean()
    long["rel_price"] = long["sell_price"] / mean_price
    long["is_promo"] = long["rel_price"] < PROMO_THRESHOLD
    long["has_event"] = long["event_type_1"].notna()
    long["event_type"] = long["event_type_1"].fillna("None")
    long["dow"] = long["date"].dt.dayofweek
    long["day_name"] = long["date"].dt.day_name().str.slice(0, 3)
    long["snap"] = long["snap_CA"].fillna(0).astype(int)
    return long


def print_summary(df: pd.DataFrame, item_id: str, store_id: str) -> None:
    """Print headline statistics used verbatim on the data and EDA slides.

    Computes the numbers the presentation quotes (span, zero-sales share,
    dispersion, promo/event coverage) and runs an augmented Dickey-Fuller test
    to justify the differencing step in ARIMA. Prints to stdout for the analyst
    to copy; produces no files.

    Args:
        df: The series frame from load_series.
        item_id: Item identifier, for the header.
        store_id: Store identifier, for the header.
    """
    n = len(df)
    zeros = (df["sales"] == 0).mean() * 100
    adf_p = adfuller(df["sales"].dropna())[1]
    print("\n" + "=" * 60)
    print(f"EDA SUMMARY  {item_id} / {store_id}")
    print("=" * 60)
    print(f"Days of history         : {n} ({df['date'].min().date()} to {df['date'].max().date()})")
    print(f"Mean daily sales        : {df['sales'].mean():.2f}")
    print(f"Std daily sales         : {df['sales'].std():.2f}")
    print(f"Coeff. of variation     : {df['sales'].std() / df['sales'].mean():.2f}")
    print(f"Zero-sales days         : {zeros:.1f}%  (quality gate: < 10%)")
    print(f"Min / Max daily sales   : {df['sales'].min():.0f} / {df['sales'].max():.0f}")
    print(f"Promo days (rel<0.95)   : {df['is_promo'].mean() * 100:.1f}%")
    print(f"Event days              : {df['has_event'].mean() * 100:.1f}%")
    print(f"ADF p-value (raw series): {adf_p:.4f}  (>0.05 => differencing needed)")
    wk = df.groupby("day_name")["sales"].mean()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"Weekend vs weekday lift : "
          f"{wk[['Sat', 'Sun']].mean() / wk[['Mon', 'Tue', 'Wed', 'Thu', 'Fri']].mean():.2f}x")
    print("Mean sales by weekday   :")
    for day in order:
        if day in wk.index:
            print(f"    {day}: {wk[day]:.2f}")
    print("=" * 60 + "\n")


def fig_timeseries(df: pd.DataFrame, out: Path) -> None:
    """Plot the full daily series with a 28-day rolling mean.

    Shows level, trend, promotional spikes, and volatility over the whole
    history. Justifies retaining outliers (spikes are real demand) and the need
    for adaptive methods. One of the figure functions called by main.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["sales"], lw=0.6, alpha=0.5, label="Daily sales")
    ax.plot(df["date"], df["sales"].rolling(28).mean(), color="crimson", lw=2,
            label="28-day rolling mean")
    ax.set_title("Daily Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_timeseries.png", dpi=160)
    plt.close(fig)


def fig_decomposition(df: pd.DataFrame, out: Path) -> None:
    """Plot an additive weekly seasonal decomposition (period=7).

    Separates trend, weekly seasonality, and residual. Justifies the seasonal
    ARIMA period m=7 and the weekly lag/calendar features. Called by main.
    """
    series = df.set_index("date")["sales"].asfreq("D").interpolate()
    result = seasonal_decompose(series, model="additive", period=7)
    fig = result.plot()
    fig.set_size_inches(14, 9)
    fig.suptitle("Seasonal Decomposition (weekly, period=7)", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "02_decomposition.png", dpi=160)
    plt.close(fig)


def fig_dow(df: pd.DataFrame, out: Path) -> None:
    """Boxplot of sales by day of week.

    Exposes the weekly demand cycle directly. Justifies day-of-week encoding
    and the lag-7 feature. Called by main.
    """
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="day_name", y="sales", order=order, ax=ax)
    ax.set_title("Demand by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("Units sold")
    fig.tight_layout()
    fig.savefig(out / "03_day_of_week.png", dpi=160)
    plt.close(fig)


def fig_acf_pacf(df: pd.DataFrame, out: Path) -> None:
    """Plot ACF and PACF out to 40 lags.

    Reveals the autocorrelation structure, in particular spikes at lag 7 and
    multiples. Justifies the lag feature set (t-1, t-7, t-14, t-28) and informs
    ARIMA order selection. Called by main.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(df["sales"].dropna(), lags=40, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")
    plot_pacf(df["sales"].dropna(), lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    fig.tight_layout()
    fig.savefig(out / "04_acf_pacf.png", dpi=160)
    plt.close(fig)


def fig_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of daily sales with the zero-sales share annotated.

    Shows the right-skewed, count-like demand distribution and the low but
    non-zero intermittency. Justifies the < 10% zero-sales quality gate and the
    emphasis on RMSE (large errors dominate inventory cost). Called by main.
    """
    zeros = (df["sales"] == 0).mean() * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(df["sales"], bins=40, kde=True, ax=ax)
    ax.axvline(df["sales"].mean(), color="crimson", ls="--", label=f"Mean {df['sales'].mean():.1f}")
    ax.set_title(f"Distribution of Daily Sales  (zero-sales days: {zeros:.1f}%)")
    ax.set_xlabel("Units sold")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_distribution.png", dpi=160)
    plt.close(fig)


def fig_price_demand(df: pd.DataFrame, out: Path) -> None:
    """Scatter of daily sales against relative price, split by promo flag.

    Visualizes price elasticity: lower relative price associates with higher
    demand. Justifies the price features and the move to models (XGBoost, TFT)
    that can use exogenous drivers ARIMA cannot. Called by main.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=df, x="rel_price", y="sales", hue="is_promo",
                    alpha=0.4, ax=ax)
    ax.axvline(PROMO_THRESHOLD, color="grey", ls=":", label="Promo threshold")
    ax.set_title("Demand vs Relative Price")
    ax.set_xlabel("Relative price (price / mean price)")
    ax.set_ylabel("Units sold")
    ax.legend(title="Promo day")
    fig.tight_layout()
    fig.savefig(out / "06_price_demand.png", dpi=160)
    plt.close(fig)


def fig_event_effect(df: pd.DataFrame, out: Path) -> None:
    """Boxplot comparing demand on event days vs non-event days.

    Quantifies the holiday/event lift. Justifies the event-type flags in the
    feature set. Called by main.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df, x="has_event", y="sales", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No event", "Event day"])
    ax.set_title("Demand on Event Days vs Normal Days")
    ax.set_xlabel("")
    ax.set_ylabel("Units sold")
    fig.tight_layout()
    fig.savefig(out / "07_event_effect.png", dpi=160)
    plt.close(fig)


def fig_rolling_volatility(df: pd.DataFrame, out: Path) -> None:
    """Plot the 28-day rolling standard deviation of demand.

    Shows that demand variance is itself non-stationary (volatility clusters
    around promotions and events). Justifies the rolling-std features and the
    forecast-error-based safety stock used by the policies. Called by main.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["sales"].rolling(28).std(), color="darkorange", lw=1.5)
    ax.set_title("28-Day Rolling Volatility of Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling std (units)")
    fig.tight_layout()
    fig.savefig(out / "08_rolling_volatility.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Parse arguments, load the series, and write all EDA figures and stats.

    Entry point. Wires the loader to every figure function and the summary
    printer, saving PNGs to the output directory for use in the thesis and
    presentation.
    """
    parser = argparse.ArgumentParser(description="EDA for one M5 product-store series.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--item", type=str, default="FOODS_3_694")
    parser.add_argument("--store", type=str, default="CA_1")
    parser.add_argument("--out", type=Path, default=Path("results/eda"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_series(args.raw_dir, args.item, args.store)
    print_summary(df, args.item, args.store)

    fig_timeseries(df, args.out)
    fig_decomposition(df, args.out)
    fig_dow(df, args.out)
    fig_acf_pacf(df, args.out)
    fig_distribution(df, args.out)
    fig_price_demand(df, args.out)
    fig_event_effect(df, args.out)
    fig_rolling_volatility(df, args.out)
    print(f"Saved 8 figures to {args.out.resolve()}")


if __name__ == "__main__":
    main()
