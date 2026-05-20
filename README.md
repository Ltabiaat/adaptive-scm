# Adaptive Supply Chain Optimization

End-to-end research system comparing demand forecasting methods (ARIMA, XGBoost, Temporal Fusion Transformer) coupled with inventory replenishment policies (EOQ, forecast-driven order-up-to, PPO reinforcement learning) on the Walmart M5 dataset.

> **Status:** Phase 0 — scaffold complete. No features implemented yet. See [`docs/PRD.md`](docs/PRD.md) for the full specification and [`CLAUDE.md`](CLAUDE.md) for Claude Code session context.

This repository accompanies the undergraduate thesis *"End-to-End AI System for Adaptive Supply Chain Optimization under Uncertainty"* (Tokyo International University, Digital Business & Innovation, Spring 2026).

---

## Quick start

### 1. Prerequisites

- Python 3.11 (`pyenv` recommended for version management)
- [`uv`](https://github.com/astral-sh/uv) for package management — install with `pip install uv` or follow the official instructions
- Git
- ~5 GB free disk space (M5 raw data + processed parquet + future model checkpoints)

### 2. Clone and install

```bash
git clone git@github.com:<your-username>/adaptive-scm.git
cd adaptive-scm

# Phase 1 only needs core deps + dev tools — fast install (~30s)
uv sync --extra dev

# Activate the virtualenv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# Sanity check
adaptive-scm status
```

When you reach Phase 2 (forecasting models), add the forecasting group:

```bash
uv sync --extra dev --extra forecasting
```

And when you reach Phase 4 (PPO), add the deep stack:

```bash
uv sync --extra dev --extra forecasting --extra deep
# or simply
uv sync --extra all
```

### 3. Get the M5 data

See [`docs/M5_DATA.md`](docs/M5_DATA.md) for download instructions. Briefly:

```bash
# After signing in to Kaggle and accepting the M5 competition rules:
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
cd data/raw && unzip m5-forecasting-accuracy.zip && cd ../..
```

The expected files in `data/raw/` after extraction:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

These are gitignored — every collaborator downloads them separately.

### 4. Run the tests

```bash
pytest tests/unit -v
```

All scaffold tests should pass. As features are implemented, more tests appear.

---

## Project layout

```
adaptive-scm/
├── config/                       # Hydra YAML configs
│   ├── default.yaml              # Base config (PRD §5)
│   ├── forecasters/              # Per-forecaster overrides
│   ├── policies/                 # Per-policy overrides
│   └── experiments/              # Per-condition overrides
├── src/adaptive_scm/             # Production code
│   ├── data/                     # M5 loading, preprocessing, features (Phase 1)
│   ├── forecasting/              # ARIMA, XGBoost, TFT (Phase 2)
│   │   └── base.py               # Forecaster ABC — DO NOT BREAK
│   ├── policies/                 # EOQ, OrderUpTo, PPO (Phases 1, 3, 4)
│   │   └── base.py               # Policy ABC — DO NOT BREAK
│   ├── simulation/               # Gym env + disruptions + runner (Phase 3)
│   ├── evaluation/               # Metrics + analysis (Phase 6)
│   ├── utils/                    # Logging, seeding
│   └── cli.py                    # Click CLI entry point
├── scripts/                      # Top-level entry points (preprocess, train, run)
├── tests/
│   ├── unit/                     # Fast, per-module tests
│   └── integration/              # End-to-end pipeline tests
├── data/
│   ├── raw/                      # M5 CSVs (gitignored)
│   └── processed/                # Parquet feature matrices (gitignored)
├── results/                      # Experiment outputs (gitignored)
├── docs/                         # PRD, M5 download guide
├── notebooks/                    # Exploration only — not production code
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

---

## Implementation roadmap

The project is phase-gated. Each phase has explicit acceptance tests (see [PRD §6](docs/PRD.md)). Do not advance phases until the previous gate is green.

| Phase | Features | Gate test |
|------:|----------|-----------|
| 0 | Scaffold + base interfaces + utils | `pytest tests/unit/test_scaffold.py` |
| 1 | Data pipeline + EOQ | `pytest tests/unit/test_data.py tests/unit/test_policies.py::test_eoq` |
| 2 | ARIMA + XGBoost + TFT | `pytest tests/unit/test_forecasting.py` |
| 3 | Gym env + OrderUpTo + disruptions | `pytest tests/unit/test_simulation.py` |
| 4 | PPO agent | `pytest tests/integration/test_pipeline.py::test_ppo_training` |
| 5 | Experiment runner + full suite | `pytest tests/integration/test_pipeline.py::test_full_run` |
| 6 | Statistical analysis | smoke test |
| 7 | Run full 30-replication suite | not a coding task |

---

## Conventions

These are non-negotiable (per PRD §2.3 / §2.4 and [`CLAUDE.md`](CLAUDE.md)):

- **Type hints** on every function signature.
- **Google-style docstrings** on every module, class, and public function. Each docstring covers (1) what it does, (2) how it works, (3) how it integrates.
- **Absolute imports only** — no relative imports.
- **No `print()`** anywhere in `src/`. Use `structlog` via `adaptive_scm.utils.logging.get_logger`.
- **No magic numbers.** All constants come from config files.
- **Line length 100**, enforced by `black`.
- **`ruff` for linting**, including pydocstyle in Google convention.

Run the toolchain locally before every commit:

```bash
ruff check src/ tests/
black src/ tests/ --check
pytest tests/unit
```

---

## Putting this on GitHub

After cloning the scaffold bundle into your local folder:

```bash
cd adaptive-scm
git init
git add .
git commit -m "Initial scaffold: project structure, config, base interfaces"
git branch -M main

# Create an empty repo on GitHub first (no README/license/.gitignore),
# then add it as the remote:
git remote add origin git@github.com:<your-username>/adaptive-scm.git
git push -u origin main
```

If you don't have SSH set up, use the HTTPS URL instead (`https://github.com/<your-username>/adaptive-scm.git`).

---

## References

Key papers driving the design — full citations live in the thesis Source Tracker, summarized in PRD §9.

- Schulman et al. (2017) — Proximal Policy Optimization
- Lim et al. (2021) — Temporal Fusion Transformers
- Chen & Guestrin (2016) — XGBoost
- Boute et al. (2022) — Deep RL for Inventory Control
- Makridakis et al. (2022) — M5 Accuracy Competition
- Theodorou et al. (2025) — Forecast Accuracy and Inventory Performance

---

## License

MIT.
