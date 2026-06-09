# adaptive-scm

Implementation for the undergraduate thesis _End-to-End AI System for Adaptive Supply Chain Optimization under Uncertainty_ (Tabiaat, TIU, 2026).

Compares three demand forecasters (ARIMA, XGBoost, TFT) crossed with three inventory policies (EOQ, forecast-driven order-up-to, PPO) on the M5 Walmart dataset, under three disruption conditions. See `docs/PRD.md` and `CLAUDE.md` for the full spec.

## Quickstart

```bash
uv sync                                       # install base deps
uv sync --extra forecasting --extra deep      # full stack incl. torch/SB3
uv sync --extra dev                           # dev tools (pytest, ruff, black)

uv run python scripts/preprocess.py           # build data/processed/<item>_<store>.parquet

uv run pytest tests/unit/                     # fast tests
uv run ruff check src/ tests/                 # lint
uv run black src/ tests/                      # format
```

## Status

- **Phase 0 — Scaffold**: ✅ complete.
- **Phase 1 — Foundation (Feature 1 + Feature 5)**: ✅ data pipeline and EOQ policy, 31 unit tests passing.
- Phase 2 — Forecasting (ARIMA, XGBoost, TFT): not started.
- Phase 3+ — Simulation, PPO, experiments, analysis: not started.

## Layout

See PRD §3.2. Production code lives in `src/adaptive_scm/`, configs in `config/`, CLI entrypoints in `scripts/`, tests in `tests/`.
