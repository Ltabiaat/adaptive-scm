# CLAUDE.md

This file provides Claude Code with persistent context for the Adaptive Supply Chain Optimization thesis project. Read this at the start of every session.

## Project Overview

This is the implementation system for Lancelot Tabiaat's undergraduate thesis at Tokyo International University, comparing classical, ML, and deep-learning forecasting methods coupled with classical and reinforcement-learning inventory policies on Walmart M5 retail data.

**Goal:** Produce empirical evidence for three thesis hypotheses by running a full experimental suite (3 forecasters × 3 policies × 3 disruption conditions × 30 replications).

**Read the full PRD before coding:** `docs/PRD.md` contains complete specifications, feature definitions, and acceptance criteria.

## Tech Stack

- Python 3.11, uv for package management
- statsmodels, pmdarima for ARIMA
- xgboost for gradient boosting
- pytorch-forecasting for TFT
- gymnasium + stable-baselines3 for PPO
- hydra-core for config, click for CLI
- pytest for testing, ruff + black for linting/formatting

## Core Conventions

- **Type hints everywhere.** Every function signature gets type annotations.
- **Google-style triple-quoted docstrings** (`"""..."""`) on every module, class, and public function. Each docstring covers three things: (1) what it does, (2) how it works (logic/formula), and (3) how it integrates with other components. Keep them short and streamlined. See PRD Section 2.4 for examples.
- **Absolute imports only.** No relative imports.
- **No print statements.** Use structlog from `src/adaptive_scm/utils/logging.py`.
- **No magic numbers.** All numeric constants come from config files.
- **Line length: 100 characters.** Enforced by black.
- **Test-first for new modules.** Write the test alongside the implementation.

## Project Structure

```
src/adaptive_scm/    # All production code lives here
├── data/            # M5 loading, preprocessing, feature engineering
├── forecasting/     # ARIMA, XGBoost, TFT (all implement Forecaster ABC)
├── policies/        # EOQ, OrderUpTo, PPO (all implement Policy ABC)
├── simulation/      # Gym environment + disruption wrappers + multi-rep runner
├── evaluation/      # Metrics and analysis
└── utils/           # Logging, seeding

config/              # Hydra YAML configs (default + overrides)
scripts/             # CLI entry points (train_*, run_*)
tests/unit/          # Unit tests, one file per module
tests/integration/   # End-to-end tests
```

## Interface Contracts (Do Not Break)

Every forecaster implements `Forecaster` ABC in `forecasting/base.py` with: `fit`, `predict`, `save`, `load`.
Every policy implements `Policy` ABC in `policies/base.py` with: `select_action`, `reset`.

If a new forecaster or policy is added, it must implement the existing interface — do not change the interface without updating all implementations and all callers.

## Commands

```bash
# Setup
uv sync                                  # Install dependencies
uv run python scripts/preprocess.py     # Preprocess M5 data

# Training
uv run python scripts/train_forecaster.py --model=arima
uv run python scripts/train_forecaster.py --model=xgboost
uv run python scripts/train_forecaster.py --model=tft
uv run python scripts/train_ppo.py --forecaster=tft

# Experiments
uv run python scripts/run_experiment.py --forecaster=tft --policy=ppo --condition=baseline
uv run python scripts/run_full_suite.py --replications=30

# Quality
uv run pytest tests/                     # All tests
uv run pytest tests/unit/                # Fast tests only
uv run ruff check src/ tests/            # Lint
uv run black src/ tests/                 # Format
```

## Development Workflow

1. **Read the PRD section** for the feature you're working on. Each feature has explicit acceptance criteria.
2. **Plan before coding.** Use planning mode for any feature with more than one file change.
3. **Vertical slices.** Build features end-to-end; don't do all-data-loading then all-forecasting then all-policies.
4. **Test as you go.** Each new function gets a unit test.
5. **Run the full pipeline test** before committing: `pytest tests/integration/test_pipeline.py`.
6. **Commit at phase gates.** See PRD Section 6 for phase boundaries.

## Out of Scope (Do Not Build)

- Multi-product or multi-store systems
- Multi-echelon networks
- Custom PPO or custom TFT implementations
- Alternative algorithms (DQN, SAC, Prophet, LSTM, etc.)
- Web UI, API endpoints, dashboards
- Database backends
- Cloud deployment

See PRD Section 1.3 for the full non-goals list.

## When You're Stuck

- Check the relevant feature in PRD Section 4 for acceptance criteria.
- Check the interface contracts in `base.py` files.
- Look at existing implementations (e.g., `arima.py`) as templates for new ones.
- Tests live alongside code; if a test exists, it specifies the contract.

## Reproducibility

Random seeds are config-driven. Never hard-code seeds. Use `utils.seeding.set_global_seed` to seed numpy, torch, and Python's random module together.
