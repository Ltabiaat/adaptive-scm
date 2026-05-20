# Product Requirements Document
## Adaptive Supply Chain Optimization System

**Author:** Lancelot Tabiaat
**Project:** Undergraduate Thesis — TIU DBI Department
**Status:** Draft v1
**Last Updated:** May 2026

---

## 1. Executive Summary

### 1.1 Project Purpose
Build an end-to-end research system that compares three demand forecasting methods (ARIMA, XGBoost, Temporal Fusion Transformer) coupled with three inventory decision policies (Economic Order Quantity, forecast-driven order-up-to, Proximal Policy Optimization reinforcement learning agent) on the M5 Walmart dataset. The system produces quantitative evidence for three hypotheses about adaptive supply chain optimization.

### 1.2 Success Criteria
The system is complete when all of the following are true:
- All three forecasters can be trained on a configured M5 product-store pair and produce 28-day forecasts.
- All three decision policies can be run against any forecaster in a simulation environment.
- The simulation supports three experimental conditions: baseline, demand spike, lead-time disruption.
- Each (forecaster, policy, condition) combination can be run for N replications and produces total cost, fill rate, holding cost, stockout frequency, service-level degradation, and recovery time.
- All hyperparameters, cost parameters, and experimental conditions are configurable via a single config file.
- The full experimental suite (3 forecasters × 3 policies × 3 conditions × 30 replications = 810 simulation runs) completes in under 24 hours on a single GPU.
- All results are persisted in a structured format suitable for analysis and plotting.

### 1.3 Non-Goals (Out of Scope)
The following are explicitly out of scope. Do not implement these:
- Multi-product or multi-store inventory systems. Single product, single store only.
- Multi-echelon supply chain networks. Single-echelon only.
- Real-time deployment or API endpoints. This is a research artifact, not a production system.
- A web UI or interactive dashboard. CLI and structured output files only.
- Alternative reinforcement learning algorithms beyond PPO (no DQN, SAC, A3C).
- Alternative forecasting models beyond the three specified (no Prophet, LSTM, N-BEATS, DeepAR).
- Hyperparameter optimization libraries beyond grid search and Optuna for XGBoost. No NAS, no AutoML.
- Custom PPO implementation. Use Stable Baselines3.
- Custom TFT implementation. Use PyTorch Forecasting.
- Database backends. Results are stored as files (Parquet/CSV).
- Cloud deployment or containerization. Local execution on macOS/Linux only.

---

## 2. Tech Stack

### 2.1 Core Stack
- **Language:** Python 3.11
- **Package manager:** uv (preferred) or pip with a requirements.txt fallback
- **Environment:** virtualenv at `.venv/`

### 2.2 Required Libraries
- **Forecasting**
  - `statsmodels >= 0.14` — ARIMA
  - `pmdarima >= 2.0` — auto.arima implementation
  - `xgboost >= 2.0` — gradient boosting
  - `pytorch-forecasting >= 1.0` — Temporal Fusion Transformer
  - `pytorch-lightning >= 2.0` — TFT training
- **Reinforcement Learning**
  - `gymnasium >= 0.29` — RL environment interface
  - `stable-baselines3 >= 2.2` — PPO implementation
  - `torch >= 2.1` — backend for SB3 and TFT
- **Data and Numerics**
  - `pandas >= 2.1`
  - `numpy >= 1.26`
  - `scikit-learn >= 1.3` — preprocessing and metrics
  - `pyarrow >= 14.0` — Parquet I/O
- **Configuration and CLI**
  - `hydra-core >= 1.3` — configuration management
  - `omegaconf >= 2.3` — config schema
  - `click >= 8.1` — CLI entry points
- **Plotting and Reporting**
  - `matplotlib >= 3.8`
  - `seaborn >= 0.13`
- **Logging**
  - `structlog >= 24.1` — structured logging
  - `tqdm >= 4.66` — progress bars

### 2.3 Conventions
- **Type hints:** Required on all function signatures.
- **Linting:** `ruff` with default rules.
- **Formatting:** `black` with line length 100.
- **Testing:** `pytest` with `pytest-cov` for coverage.
- **Imports:** Absolute imports only. No relative imports.
- **No print statements:** Use the structured logger.
- **No magic numbers:** All numeric constants live in the config file or are derived from it.

### 2.4 Documentation Standards
All code must be documented using Python triple-quoted docstrings (`"""..."""`) in **Google style**. Docstrings are not optional — they are a hard requirement enforced during code review. Keep them short and streamlined; clarity beats verbosity.

**Required for every module** (top of file):
```python
"""One-line module summary.

Brief description of the module's purpose and what it integrates with.
"""
```

**Required for every public function and method:**
```python
def compute_safety_stock(sigma_lead: float, lead_time: int, z: float) -> float:
    """Compute safety stock for a given service level.

    Uses the standard formula ss = z * sigma * sqrt(L). Called by the EOQ
    and order-up-to policies during their target-level computation.

    Args:
        sigma_lead: Standard deviation of forecast errors over the lead time.
        lead_time: Lead time in days.
        z: Service-level z-score (e.g., 1.645 for 95% cycle service).

    Returns:
        Safety stock quantity in units.
    """
```

**Required for every class:**
```python
class PPOAgent(Policy):
    """Proximal Policy Optimization agent for inventory replenishment.

    Wraps Stable Baselines3's PPO implementation and exposes it through the
    Policy interface so it is interchangeable with EOQ and OrderUpTo in the
    simulation. The agent consumes forecast features in its state vector
    (see Feature 9 in PRD).
    """
```

**Three things every docstring must cover:**
1. **What it does** — one-line summary of purpose.
2. **How it works** — brief note on the logic, formula, or algorithm used.
3. **How it integrates** — which other components call it or are called by it.

**Inline comments** (`#`) are used sparingly and only when the code itself cannot make the intent obvious. Prefer renaming variables and extracting functions over adding comments.

---

## 3. System Architecture

### 3.1 High-Level Pipeline
```
[M5 Raw Data] → [Data Loader] → [Preprocessor] → [Feature Engineer]
                                                          ↓
                                              [Forecasting Module]
                                              (ARIMA / XGBoost / TFT)
                                                          ↓
                                                  [Forecast Output]
                                                          ↓
                                              [Decision Module]
                                       (EOQ / Order-Up-To / PPO)
                                                          ↓
                                                  [Order Decisions]
                                                          ↓
                                              [Simulation Environment]
                                                          ↓
                                              [Metrics Collector]
                                                          ↓
                                              [Results Persistence]
```

### 3.2 Directory Structure
```
adaptive-scm/
├── config/
│   ├── default.yaml              # Base config
│   ├── forecasters/
│   │   ├── arima.yaml
│   │   ├── xgboost.yaml
│   │   └── tft.yaml
│   ├── policies/
│   │   ├── eoq.yaml
│   │   ├── order_up_to.yaml
│   │   └── ppo.yaml
│   └── experiments/
│       ├── baseline.yaml
│       ├── demand_spike.yaml
│       └── lead_time_disruption.yaml
├── src/
│   ├── adaptive_scm/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py         # M5 data loading
│   │   │   ├── preprocessor.py   # Cleaning, normalization
│   │   │   └── features.py       # Feature engineering
│   │   ├── forecasting/
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Abstract Forecaster interface
│   │   │   ├── arima.py
│   │   │   ├── xgboost.py
│   │   │   └── tft.py
│   │   ├── policies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # Abstract Policy interface
│   │   │   ├── eoq.py
│   │   │   ├── order_up_to.py
│   │   │   └── ppo.py
│   │   ├── simulation/
│   │   │   ├── __init__.py
│   │   │   ├── environment.py    # Gym env
│   │   │   ├── disruptions.py    # Demand/lead-time disruption scenarios
│   │   │   └── runner.py         # Multi-replication runner
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py        # RMSE, MAPE, fill rate, etc.
│   │   │   └── analyzer.py       # Aggregate analysis across runs
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   └── seeding.py        # Reproducibility
│   │   └── cli.py                # Click entry points
├── scripts/
│   ├── train_forecaster.py       # Train one forecaster
│   ├── train_ppo.py              # Train PPO agent
│   ├── run_experiment.py         # Run one (forecaster, policy, condition)
│   └── run_full_suite.py         # Run all 810 combinations
├── tests/
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_forecasting.py
│   │   ├── test_policies.py
│   │   └── test_simulation.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_full_run.py
├── data/                          # Raw and processed data (gitignored)
│   ├── raw/                       # M5 files
│   └── processed/                 # Cleaned, feature-engineered
├── results/                       # Experiment outputs (gitignored)
│   ├── forecasts/                 # Per-forecaster outputs
│   ├── simulations/               # Per-run simulation logs
│   └── analysis/                  # Aggregated results
├── notebooks/                     # Exploration only, not production
├── CLAUDE.md                      # Claude Code context (see separate file)
├── pyproject.toml
├── README.md
└── .gitignore
```

### 3.3 Interface Contracts
Every module exposes a small set of clean interfaces. Implementations may vary; interfaces must not.

**Forecaster Interface (`src/adaptive_scm/forecasting/base.py`)**
```python
class Forecaster(ABC):
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """Train on historical data."""

    @abstractmethod
    def predict(self, horizon: int) -> ForecastOutput:
        """Generate forecast for next N days."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster": ...
```

**ForecastOutput dataclass**
```python
@dataclass
class ForecastOutput:
    point_forecast: np.ndarray         # shape (horizon,)
    lower_bound: np.ndarray | None     # P10 for TFT, mean-1.96σ for ARIMA/XGB
    upper_bound: np.ndarray | None     # P90 for TFT, mean+1.96σ for ARIMA/XGB
    historical_rmse: float             # for noise calibration in simulation
```

**Policy Interface (`src/adaptive_scm/policies/base.py`)**
```python
class Policy(ABC):
    @abstractmethod
    def select_action(self, state: State) -> int:
        """Return order quantity for current period."""

    @abstractmethod
    def reset(self) -> None: ...
```

---

## 4. Feature Specifications

Build features in the following order. Each feature is a **vertical slice** — it works end-to-end before the next feature starts. Each feature has a testable acceptance criterion.

### Feature 1: Data Loading and Preprocessing
**Priority:** P0 (foundation)
**Dependencies:** None
**Output:** A clean, feature-engineered DataFrame for a configured product-store pair.

**Functional requirements:**
- Load M5 raw files from `data/raw/` (sales_train_evaluation.csv, calendar.csv, sell_prices.csv).
- Select a single product-store pair via config (e.g., `item_id=FOODS_3_090`, `store_id=CA_1`).
- Validate the selected series has < 10% zero-sales days, ≥ 4 years of history, and ≥ 1 promotional event per year. Raise a clear error if not.
- Forward-fill missing prices.
- Normalize prices to a relative index (price / mean price).
- Generate engineered features for ML/DL models:
  - Lag features: t-1, t-7, t-14, t-28, t-365
  - Rolling stats: 7-day and 28-day rolling mean and std of sales
  - Calendar features: day-of-week (one-hot), day-of-month, week-of-year, month, start/end-of-month flags
  - Event indicators: four binary flags for M5 event types, plus one-hot specific event
  - Price features: relative price index, weekly price change, promotional flag (price < 0.95)
- Time-ordered split: 1,597 train / 28 validation / 28 test.
- Persist processed DataFrame to `data/processed/{item_id}_{store_id}.parquet`.

**Acceptance criteria:**
- `pytest tests/unit/test_data.py` passes.
- `python scripts/preprocess.py` produces a Parquet file for the configured product-store pair.
- The output DataFrame has columns: `date`, `sales`, `split` (train/val/test), and all engineered features.

### Feature 2: ARIMA Forecaster
**Priority:** P0
**Dependencies:** Feature 1
**Output:** A trained ARIMA model that produces 28-day point forecasts.

**Functional requirements:**
- Implements the `Forecaster` interface.
- Uses `pmdarima.auto_arima` for order selection.
- Searches over (p, d, q) and seasonal (P, D, Q) with weekly seasonality (m=7).
- Minimizes AIC.
- Fits on the 1,597-day training period only.
- Produces 28-day point forecasts with 95% confidence intervals.
- Reports historical RMSE on the validation set for downstream noise calibration.

**Acceptance criteria:**
- `pytest tests/unit/test_forecasting.py::test_arima` passes.
- A trained ARIMA can be saved and loaded round-trip without loss of state.
- On the test product, ARIMA's RMSE is logged and within an order of magnitude of the M5 published statistical benchmarks (sanity check, not strict target).

### Feature 3: XGBoost Forecaster
**Priority:** P0
**Dependencies:** Feature 1
**Output:** A trained XGBoost model that produces 28-day recursive forecasts.

**Functional requirements:**
- Implements the `Forecaster` interface.
- Uses the full engineered feature set from Feature 1.
- Target variable: daily unit sales.
- Hyperparameter grid:
  - tree depth: [3, 6, 9]
  - learning rate: [0.01, 0.05, 0.1]
  - n_estimators: [100, 300, 500]
  - L2 regularization (lambda): [0, 0.1, 1.0]
- Grid search on the validation set with early stopping (patience=20 on validation RMSE).
- 28-day forecast generated recursively: predict day t, use as lag for day t+1.
- Confidence intervals approximated as point ± 1.96 × historical_rmse.

**Acceptance criteria:**
- `pytest tests/unit/test_forecasting.py::test_xgboost` passes.
- The best hyperparameters from grid search are logged and persisted.
- On the test product, XGBoost outperforms the seasonal naive baseline (sales at t-7) on validation RMSE.

### Feature 4: TFT Forecaster
**Priority:** P0
**Dependencies:** Feature 1
**Output:** A trained TFT model that produces 28-day probabilistic forecasts.

**Functional requirements:**
- Implements the `Forecaster` interface.
- Uses PyTorch Forecasting's reference TFT implementation.
- Inputs partitioned correctly:
  - Static categoricals: item_id, store_id, dept_id, cat_id
  - Time-varying known: prices, calendar features, event flags
  - Time-varying unknown: historical sales
- Training:
  - Adam optimizer, learning rate 1e-3
  - Batch size 64
  - Quantile loss targeting P10, P50, P90
  - Early stopping with patience 10 epochs on validation quantile loss
  - Maximum 50 epochs
- Produces 28-day probabilistic forecast in single forward pass (not recursive).
- ForecastOutput returns P50 as point_forecast, P10 as lower_bound, P90 as upper_bound.

**Acceptance criteria:**
- `pytest tests/unit/test_forecasting.py::test_tft` passes.
- Training converges within 50 epochs on the test product.
- ForecastOutput contains P10/P50/P90 as expected.

### Feature 5: EOQ Baseline Policy
**Priority:** P0
**Dependencies:** Feature 1 (for forecast input)
**Output:** A policy that places orders based on the EOQ formula plus safety stock.

**Functional requirements:**
- Implements the `Policy` interface.
- Computes Q* = sqrt(2 × D × S / H), where D is annualized forecast demand, S is fixed order cost, H is annual holding cost.
- Safety stock: ss = 1.645 × σ_L × sqrt(L), where σ_L is forecast error std over lead time.
- Reorder point: ROP = mean_daily_demand × L + ss.
- Places order of size Q* when inventory position ≤ ROP.
- Reads cost parameters and lead time from config.

**Acceptance criteria:**
- `pytest tests/unit/test_policies.py::test_eoq` passes.
- Given a fixed forecast and inventory state, EOQ returns deterministic order quantities.

### Feature 6: Order-Up-To Baseline Policy
**Priority:** P0
**Dependencies:** Features 1, 2 (forecast input)
**Output:** A forecast-driven dynamic base-stock policy.

**Functional requirements:**
- Implements the `Policy` interface.
- At each decision epoch:
  - Compute target level S_t = sum(forecast over R+L days) + 1.645 × σ_{R+L}
  - Order quantity a_t = max(0, S_t - inventory_position)
- R is the review period (config, default 1 day).
- L is the lead time (config).
- σ_{R+L} is forecast error std over the protection interval.
- Uses the same forecast information as PPO.

**Acceptance criteria:**
- `pytest tests/unit/test_policies.py::test_order_up_to` passes.
- Order quantities respond to changes in forecast and inventory position.

### Feature 7: Simulation Environment
**Priority:** P0
**Dependencies:** Features 5, 6 (to test with classical policies first)
**Output:** A Gymnasium-compatible inventory environment.

**Functional requirements:**
- Single-echelon, single-product, daily-step simulation.
- State vector (for PPO; classical policies use a subset):
  - On-hand inventory (float)
  - Pipeline orders by remaining lead time (vector, length = max_lead_time)
  - 7-day forecast point estimate (vector, length 7)
  - 7-day forecast uncertainty (vector, length 7)
  - Day-of-week (one-hot, length 7)
  - Upcoming event flags (binary, length 7)
- Action space (PPO): Discrete(11), corresponding to multiples of mean daily demand: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] × d_bar.
- Reward: r_t = -(h × max(inv, 0) + p × max(-inv, 0) + K × I[a > 0] + c × a)
- Cost parameters from config: h=0.05, p=2.00, K=10.00, c=1.00 (defaults; configurable).
- Lead time: base (config, default 3) + uniform(0, 2) stochastic.
- Demand realization: forecast_value × lognormal_noise(0, forecast_rmse).
- Episode length: 365 days for training, 28 days for evaluation.
- Stockouts are lost sales, not backordered.

**Acceptance criteria:**
- `pytest tests/unit/test_simulation.py` passes.
- The environment passes `gymnasium.utils.env_checker.check_env`.
- Running EOQ for 28 days in the env produces a valid trajectory.

### Feature 8: Disruption Scenarios
**Priority:** P0
**Dependencies:** Feature 7
**Output:** Two disruption modifiers that can be applied to the environment.

**Functional requirements:**
- **Demand spike:** During a configurable 14-day window, multiply realized demand by 1.5. Window start day is configurable.
- **Lead-time disruption:** During a configurable 14-day window, double the base lead time. Window start day is configurable.
- Disruptions are applied via env wrappers, not by modifying the core env code.
- The baseline (no disruption) is also a valid configuration.

**Acceptance criteria:**
- `pytest tests/unit/test_simulation.py::test_disruptions` passes.
- Applying each disruption produces visible changes in the demand or lead-time trajectory at the configured window.

### Feature 9: PPO Agent
**Priority:** P0
**Dependencies:** Feature 7
**Output:** A trained PPO agent that can be evaluated in the simulation environment.

**Functional requirements:**
- Uses `stable_baselines3.PPO` with `MlpPolicy`.
- Hyperparameters from config (defaults below):
  - clip_range (ε): 0.2
  - gamma: 0.99
  - gae_lambda: 0.95
  - learning_rate: 3e-4
  - batch_size: 64
  - n_epochs: 4
  - n_steps: 2048
  - policy network: [64, 64] with tanh activation
- Training: 500,000 environment steps.
- Episodes randomized across the training period (different start dates).
- Forecast in state is generated by a frozen forecaster (loaded from disk; not retrained during PPO training).
- Saves trained agent to `results/ppo_{forecaster_name}.zip`.

**Acceptance criteria:**
- `pytest tests/integration/test_pipeline.py::test_ppo_training` passes (with reduced step count, e.g., 5000, for speed).
- Training loss curve decreases monotonically (with smoothing) over the training period.
- A trained PPO can be loaded and produces valid actions for any state.

### Feature 10: Experiment Runner
**Priority:** P0
**Dependencies:** Features 1-9
**Output:** A CLI script that runs one (forecaster, policy, condition) experiment for N replications and saves results.

**Functional requirements:**
- `python scripts/run_experiment.py --forecaster=arima --policy=eoq --condition=baseline --replications=30`
- For each replication:
  - Set a unique random seed.
  - Load the trained forecaster.
  - Initialize the simulation environment with the specified condition.
  - Run the specified policy for 28 days.
  - Record per-day: demand, order, inventory, holding cost, stockout cost, reward.
- Aggregate metrics across replications: total cost (mean, std), fill rate, holding cost, stockout frequency.
- Resilience metrics under disruption: service-level degradation, recovery time.
- Save to `results/simulations/{forecaster}_{policy}_{condition}.parquet`.

**Acceptance criteria:**
- A single-replication run completes in under 30 seconds.
- Output file contains one row per (replication, day) plus a summary row of aggregates.

### Feature 11: Full Experimental Suite
**Priority:** P0
**Dependencies:** Feature 10
**Output:** A script that runs all 3 × 3 × 3 = 27 (forecaster, policy, condition) combinations and aggregates results.

**Functional requirements:**
- `python scripts/run_full_suite.py --replications=30`
- Runs all 27 combinations with the configured replication count.
- Skips combinations whose results file already exists (resumable).
- Aggregates all results into `results/analysis/full_suite.parquet`.
- Generates a Markdown summary report at `results/analysis/summary.md` with tables of all metrics by (forecaster, policy, condition).

**Acceptance criteria:**
- Full suite with 30 replications completes in under 24 hours on a single GPU.
- Summary report contains: total cost table, fill rate table, resilience metrics table, and correlation between forecast RMSE and inventory cost (for H3).

### Feature 12: Analysis and Hypothesis Testing
**Priority:** P1 (nice-to-have but valuable)
**Dependencies:** Feature 11
**Output:** Statistical analysis script that produces hypothesis test results.

**Functional requirements:**
- For H1 (PPO > classical policies): Paired t-tests comparing PPO total cost vs EOQ total cost and PPO vs order-up-to, holding forecaster constant.
- For H2 (integrated > standalone): Comparison of TFT+PPO vs each of the 6 baseline combinations.
- For H3 (accuracy ≠ decision quality): Spearman rank correlation between forecast RMSE and inventory cost across all (forecaster, policy) combinations.
- Output: `results/analysis/hypothesis_tests.md` with p-values, effect sizes, and interpretation.

**Acceptance criteria:**
- Report contains numerical evidence for or against each hypothesis.
- Report can be opened in any Markdown viewer.

---

## 5. Configuration Schema

All configurable parameters live in `config/default.yaml` with overrides per forecaster/policy/experiment.

```yaml
# config/default.yaml

data:
  raw_dir: data/raw
  processed_dir: data/processed
  product_store:
    item_id: FOODS_3_090
    store_id: CA_1
  splits:
    train_days: 1597
    val_days: 28
    test_days: 28

simulation:
  costs:
    holding_per_unit_per_day: 0.05
    stockout_per_unit: 2.00
    fixed_order: 10.00
    purchase_per_unit: 1.00
  lead_time:
    base: 3
    stochastic_max_additional: 2
  episode:
    length: 28              # test horizon
    training_length: 365    # PPO training episodes
  noise:
    distribution: lognormal

experiments:
  conditions:
    - baseline
    - demand_spike
    - lead_time_disruption
  disruption_window:
    start_day: 7
    duration_days: 14
  demand_spike_multiplier: 1.5
  lead_time_disruption_multiplier: 2.0
  replications: 30
  random_seeds: [42, 43, 44, ...]   # explicit list for reproducibility

forecasters:
  arima:
    seasonal: true
    seasonal_period: 7
    information_criterion: aic

  xgboost:
    grid_search:
      max_depth: [3, 6, 9]
      learning_rate: [0.01, 0.05, 0.1]
      n_estimators: [100, 300, 500]
      reg_lambda: [0, 0.1, 1.0]
    early_stopping_rounds: 20

  tft:
    learning_rate: 0.001
    batch_size: 64
    max_epochs: 50
    early_stopping_patience: 10
    quantiles: [0.1, 0.5, 0.9]
    hidden_size: 16
    attention_head_size: 4

policies:
  eoq:
    service_level: 0.95

  order_up_to:
    review_period: 1
    service_level: 0.95

  ppo:
    total_timesteps: 500000
    clip_range: 0.2
    gamma: 0.99
    gae_lambda: 0.95
    learning_rate: 0.0003
    batch_size: 64
    n_epochs: 4
    n_steps: 2048
    policy_net: [64, 64]
    activation: tanh
```

---

## 6. Implementation Plan (Phase-Gated)

Each phase has a clear gate: all tests pass and acceptance criteria are met before moving to the next phase.

### Phase 1: Foundation (Features 1, 5)
**Gate:** Data loading works end-to-end; EOQ policy passes unit tests against a mocked forecast.
**Tests required:** `test_data.py`, `test_policies.py::test_eoq`

### Phase 2: Forecasting Models (Features 2, 3, 4)
**Gate:** All three forecasters can be trained and produce valid 28-day forecasts on the test product.
**Tests required:** `test_forecasting.py` (all three model tests)

### Phase 3: Simulation Core (Features 6, 7, 8)
**Gate:** Simulation runs cleanly with EOQ and order-up-to policies under all three conditions.
**Tests required:** `test_simulation.py`, `test_policies.py::test_order_up_to`

### Phase 4: PPO Integration (Feature 9)
**Gate:** PPO trains successfully on a reduced timestep budget; loaded agent produces valid actions.
**Tests required:** `test_pipeline.py::test_ppo_training` (with 5000-step training for CI)

### Phase 5: Experiment Orchestration (Features 10, 11)
**Gate:** Full suite runs end-to-end with reduced replications (e.g., 3) and produces summary report.
**Tests required:** `test_pipeline.py::test_full_run`

### Phase 6: Analysis (Feature 12)
**Gate:** Hypothesis test report generated from suite results.
**Tests required:** Smoke test that the analysis script runs without error.

### Phase 7: Full Experimental Run
**Not a coding task.** Run the full 30-replication suite for thesis results.

---

## 7. Testing Strategy

### 7.1 Unit Tests
- Every module has a corresponding test file in `tests/unit/`.
- Aim for ≥ 70% code coverage on `src/`.
- Mock external dependencies (file I/O, model training) where possible to keep tests fast.

### 7.2 Integration Tests
- `tests/integration/test_pipeline.py` exercises full forecaster + policy + simulation flow with small data.
- These tests may take longer (up to 5 minutes total) and are not run on every commit.

### 7.3 Reproducibility Tests
- Given a fixed random seed, the full pipeline produces identical results across runs.
- A regression test compares stored "golden" outputs against fresh runs.

---

## 8. Open Questions and Risks

These items should be flagged early during implementation; they are not blockers but require attention.

- **GPU availability:** TFT training and PPO training are GPU-bound. The implementation must run on CPU as a fallback (slower but correct).
- **Convergence of PPO:** PPO may not converge cleanly on the inventory problem in 500k steps. The implementation should log training curves and surface lack-of-convergence as a warning, not a silent failure.
- **Forecast horizon mismatch:** PPO state uses a 7-day forecast window, but forecasters produce 28-day forecasts. The implementation should slice the 28-day forecast to the relevant 7-day window inside the env, not require forecasters to support a 7-day mode.
- **Reward scaling:** If the cost magnitudes produce very large negative rewards, PPO may struggle. Consider reward normalization via `VecNormalize` from Stable Baselines3.

---

## 9. References

- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv.
- Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting. International Journal of Forecasting.
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Boute et al. (2022). Deep Reinforcement Learning for Inventory Control: A Roadmap. EJOR.
- Makridakis et al. (2022). M5 Accuracy Competition: Results, Findings, Conclusions. International Journal of Forecasting.
- Theodorou et al. (2025). Forecast Accuracy and Inventory Performance. EJOR.

See full citations in the thesis Source Tracker.
