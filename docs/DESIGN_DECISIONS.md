# Design Decisions Log

A running record of interpretation choices and deviations from `PRD.md`, kept so
they can be reviewed, defended in the thesis, or reversed later. Each entry notes
**what** was decided, **why**, and **how to change it** if you disagree.

Status legend: 🟢 low-risk / cosmetic · 🟡 worth a look · 🔴 affects results, review before the full run.

---

## Cross-cutting (scaffold)

### D-0.1 Optional dependency groups instead of a flat requirements list 🟢
- **What:** Forecasting (`statsmodels`, `pmdarima`, `xgboost`) and deep-learning
  (`torch`, `pytorch-forecasting`, `stable-baselines3`, `gymnasium`) libraries live
  in optional extras (`forecasting`, `deep`, `dev`) rather than the base install.
- **Why:** The data + classical-policy layers don't need the heavy stack; keeps a
  base install fast and lets CI test early phases without GPU-tier dependencies.
- **Change it:** Move the extras back into `[project.dependencies]` in `pyproject.toml`.

### D-0.2 Loosened Python pin (`>=3.11`) 🟢
- **What:** Repo targets `>=3.11,<3.12`; relaxed locally where needed to run in a 3.12 container.
- **Why:** Environment portability during development. The committed pin stays at the PRD value.
- **Change it:** Nothing to change in the repo; just be aware test runs may use 3.12.

---

## Phase 1 — Data pipeline (Feature 1) & EOQ (Feature 5)

### D-1.1 `σ_L` read as the *daily* forecast-error SD 🔴
- **What:** PRD writes safety stock as `ss = 1.645 · σ_L · √L`. We interpret `σ_L`
  as the **daily** forecast-error SD (proxied by the forecaster's `historical_rmse`),
  so the implemented formula is `ss = z · σ_d · √L` — the standard textbook form.
- **Why:** Taking `σ_L` literally as "SD over the lead time" and then multiplying by
  `√L` again would double-count the lead-time scaling. The textbook form is the
  defensible one.
- **Change it:** In `policies/eoq.py::_reorder_point`, drop the `√L` (use
  `z · σ_L`) if you want the literal PRD reading. **Note:** the order-up-to policy
  (Feature 6) faces the same `σ_{R+L}` question — keep the two consistent.

### D-1.2 "≥1 promotional event per year" gate = non-null `event_name_1` per calendar year 🟡
- **What:** The Feature 1 validation gate is implemented as "at least one non-null
  M5 calendar event (`event_name_1`) in each calendar year of history."
- **Why:** M5's "events" are calendar holidays, not promotions; the only promo signal
  is the price-based `is_promo` flag, which is computed *after* validation. The literal
  reading that survives is "has calendar events each year," which also catches
  truncated series with stripped event columns.
- **Change it:** In `data/loader.py::validate_series`, swap the event check for a
  price-based one (e.g. `is_promo` rate > 0 per year) if you meant promotions.

### D-1.3 Event one-hot uses only `event_name_1` 🟡
- **What:** Specific-event one-hot encoding uses `event_name_1` only; the secondary
  `event_name_2` slot is dropped.
- **Why:** `event_name_2` is extremely sparse in M5 and overlaps the primary slot;
  including it adds many near-empty columns for little signal.
- **Change it:** In `data/features.py::_add_event_features`, add a second
  `get_dummies` on `event_name_2` and union the columns.

### D-1.4 Rolling features shifted by 1 day (no leakage) 🟢
- **What:** `sales_roll_mean/std_{7,28}` at day *t* summarize days strictly before *t*.
- **Why:** Prevents the target day's own sales leaking into its features.
- **Change it:** Remove the `.shift(1)` in `_add_rolling_features` (not recommended).

### D-1.5 EOQ uses the lead-time window of the forecast for mean demand 🟡
- **What:** Mean daily demand for both `Q*` (after annualizing) and the reorder point
  is the mean of the **first L days** of the point forecast.
- **Why:** The reorder decision concerns the lead-time horizon, so demand over that
  window is the relevant estimate.
- **Change it:** In `policies/eoq.py::_mean_daily_demand`, average the full horizon
  instead of the first `L` days.

---

## Phase 2 — ARIMA forecaster (Feature 2)

### D-2.1 `fit()` receives the full split frame; trains on `train`, scores on `val` 🟡
- **What:** Every forecaster's `fit(train_data)` takes the full preprocessed frame
  (with the `split` column). It fits on `split == "train"` and computes
  `historical_rmse` on `split == "val"`. Falls back to in-sample residual RMSE if
  there is no `val` split.
- **Why:** Satisfies "fit on train, report RMSE on val" without changing the
  `Forecaster` ABC signature, and matches what XGBoost needs (val set for early
  stopping) and TFT (val set for quantile-loss early stopping).
- **Change it:** Add an explicit `val_data` argument to `Forecaster.fit` (ABC change,
  touches all three forecasters and their callers).

### D-2.2 Forecasts floored at zero 🟡
- **What:** Point forecast and both CI bounds are clipped to `>= 0` in `predict()`
  (and in the RMSE computation), for every forecaster.
- **Why:** The forecast feeds an inventory simulator where demand cannot be negative;
  ARIMA in particular can emit negative points for low-volume days.
- **Change it:** Remove the `np.clip(..., 0, None)` calls in the forecaster and floor
  only at the simulator boundary instead.

### D-2.3 Additive ARIMA search-bound config knobs + start-order clamp 🟢
- **What:** `config/forecasters/arima.yaml` adds `max_p/max_q/max_P/max_Q/max_d/max_D/
  stepwise` (defaults mirror `auto_arima`). `fit()` also clamps `start_p/start_q` down
  to the configured max so tight test bounds don't trip pmdarima's `max >= start` check.
- **Why:** Lets unit tests constrain the order search to run in seconds; the PRD config
  only specified `seasonal/seasonal_period/information_criterion`.
- **Change it:** Delete the extra keys; they default to `auto_arima`'s own values.

---

## Phase 3 — XGBoost forecaster (Feature 3)

### D-3.1 Model feature set = engineered columns only (raw passthrough excluded) 🟡
- **What:** `data/features.py::feature_columns` returns the engineered features
  (lags, rolling, calendar one-hots, event flags, price features) and **excludes**
  raw passthrough columns: `date, d, id columns, sales, split, wm_yr_wk, wday, month,
  year, sell_price, snap`, and the raw event name/type strings.
- **Why:** Keeps the model feature set faithful to the PRD Feature 1 list and avoids
  duplicate signals (`month` vs `month_num`, `sell_price` vs `price_index`). `snap`
  is excluded because it is not in the PRD feature list.
- **Change it:** Edit `NON_FEATURE_COLUMNS` in `data/features.py` (e.g. drop `snap`
  from the exclusion set to let the model use SNAP days).

### D-3.2 Recursive forecasting overwrites sales-derived features from a prediction buffer 🔴
- **What:** For the multi-step forecast, lag/rolling features (`sales_lag_*`,
  `sales_roll_*`) are recomputed at each step from a buffer of
  `[training-sales tail … predictions so far]`; known future covariates (calendar,
  event, price) come from stored future rows. Predictions feed back as lags for later
  steps (PRD "recursive" requirement).
- **Why:** A true out-of-sample multi-step forecast cannot use the future actuals that
  the precomputed lag columns contain (that would be leakage). Rolling std uses
  `ddof=1` to match pandas' rolling default.
- **Change it:** This is intrinsic to recursive forecasting; the alternative is a
  direct multi-output model, which the PRD does not specify.

### D-3.3 `fit()` stashes future covariate rows + training-sales buffer 🟡
- **What:** XGBoost `fit()` stores the post-train rows (val + test, already engineered)
  and the training sales array so `predict(horizon)` can build recursive features
  without extra arguments. `predict(horizon)` forecasts the `horizon` days immediately
  after training, using the first `horizon` stored future rows.
- **Why:** Keeps the `predict(horizon)` signature unchanged (the ABC takes only an int)
  while supplying the known future covariates XGBoost/TFT need.
- **Change it:** Add a `future_covariates` argument to `predict` (ABC change).

### D-3.4 Grid selection on one-step val RMSE; `historical_rmse` is recursive val RMSE 🔴
- **What:** The 81-point grid (`max_depth × learning_rate × n_estimators × reg_lambda`)
  is selected by **one-step** validation RMSE (each val day predicted from its actual
  features). Early stopping (patience 20) monitors the same one-step val metric.
  `historical_rmse` (used for CIs and simulator noise) is instead the **recursive**
  multi-step RMSE of the selected model over the val horizon.
- **Why:** One-step RMSE is the standard, cheap selection/early-stopping metric and is
  the natural basis for the "beats seasonal-naive" check. `historical_rmse` should
  reflect the *deployed* multi-step forecast error, so it is computed recursively —
  consistent with how ARIMA's `historical_rmse` is measured over its 28-day forecast.
  Both numbers are logged.
- **Change it:** In `forecasting/xgboost.py`, select on recursive RMSE too (one metric)
  if you prefer selection and reporting to use the same number — costs ~81 extra cheap
  recursive forecasts.

### D-3.5 CIs as point ± 1.96 · `historical_rmse`, floored at zero 🟢
- **What:** XGBoost has no native predictive interval; bounds are
  `point ± 1.96 · historical_rmse`, clipped to `>= 0` (per D-2.2).
- **Why:** PRD Feature 3 specifies exactly this approximation.
- **Change it:** Swap in quantile regression (`reg:quantileerror`) for empirical bounds.

### D-3.6 Seasonal-naive baseline for the acceptance check = sales at t−7, one-step 🟢
- **What:** The "outperforms seasonal naive" criterion compares the model's one-step
  val RMSE against a `sales[t-7]` naive forecast over the same val days.
- **Why:** Matches the one-step selection metric (D-3.4) for an apples-to-apples check.
- **Change it:** Compare recursive forecasts instead if you change D-3.4.

---

## Phase 4 — TFT forecaster (Feature 4)

### D-4.1 Sales-derived features excluded from TFT inputs 🔴
- **What:** TFT consumes raw ``sales`` as its time-varying *unknown* plus the
  calendar/event/price features as time-varying *known* reals. The engineered
  ``sales_lag_*`` and ``sales_roll_*`` columns are **excluded** entirely.
- **Why:** In the decoder (forecast) window those columns contain *future
  actuals* — e.g. ``sales_lag_1`` of day t+2 is day t+1's actual sales — so
  passing them as "known" covariates would leak the test answer. TFT learns
  the autoregressive structure through its encoder from raw sales instead,
  which is exactly the PRD's input partitioning ("time-varying unknown:
  historical sales").
- **Change it:** Don't. If lag features are ever wanted, they must be moved to
  the *unknown* group (where the decoder cannot see them), which adds nothing
  beyond the raw series.

### D-4.2 Quantile crossing repaired by sorting 🟡
- **What:** Predicted quantiles are sorted along the quantile axis before
  mapping P10/P50/P90 to lower/point/upper, then floored at zero (D-2.2).
- **Why:** Quantile-loss models can emit crossed quantiles (P10 > P50),
  especially early in training; pytorch-forecasting does not enforce
  monotonicity. Sorting is the standard non-crossing repair and is a no-op
  when quantiles are already ordered.
- **Change it:** Remove the ``np.sort`` in ``predict`` and assert ordering
  instead, if you'd rather treat crossing as a training failure.

### D-4.3 Additive ``encoder_length`` config knob (default 56) 🟡
- **What:** The PRD does not specify the encoder window. Added
  ``encoder_length`` to the tft config block, default 56 days (8 weeks = 2×
  the 28-day horizon).
- **Why:** The encoder window is a required TimeSeriesDataSet parameter; 2×
  horizon covering several weekly cycles is a common, defensible default.
- **Change it:** Edit ``config/forecasters/tft.yaml``; it's purely config.

### D-4.4 TFT saves a directory (checkpoint + metadata) 🟢
- **What:** ``save(path)`` writes a directory containing ``model.ckpt``
  (Lightning checkpoint) and ``meta.joblib`` (stored frame, dataset
  parameters, RMSE), unlike ARIMA/XGBoost's single joblib file. ``save``
  requires the originally fitted instance (a loaded instance cannot re-save).
- **Why:** Lightning's checkpoint format is the robust way to round-trip a
  TFT; the metadata sidecar carries what the checkpoint can't. The
  ``Forecaster`` ABC explicitly allows directory artifacts. Re-saving a
  loaded model is not needed in the train-once / evaluate-many workflow.
- **Change it:** Bundle both into one ``torch.save`` payload if a single file
  is ever required.

### D-4.5 ``historical_rmse`` = P50 vs validation actuals 🟢
- **What:** TFT's ``historical_rmse`` is the RMSE of its P50 forecast over the
  28-day validation horizon, computed via the same single-pass ``predict``
  path used at evaluation time.
- **Why:** Consistent with D-2.1/D-3.4: every forecaster's ``historical_rmse``
  measures the *deployed* multi-step forecast against held-out validation data.
- **Change it:** Nothing to change unless D-2.1 changes.

### D-4.6 TFT trainer defaults to CPU; accelerator is a config knob 🟡
- **What:** The Lightning trainer in both ``fit`` and ``predict`` uses
  ``accelerator: cpu`` by default, configurable via
  ``forecasters.tft.accelerator``. Previously ``"auto"``.
- **Why:** ``auto`` selects Apple's MPS backend on M-series Macs, where
  pytorch-forecasting's TFT hangs (observed: unit test froze indefinitely at
  the first TFT fit). CPU is correct everywhere and matches the PRD's
  "must run on CPU as a fallback" requirement. On a CUDA machine, set
  ``accelerator: gpu`` (or ``auto``) in config for the real training runs.
- **Change it:** Edit ``config/forecasters/tft.yaml``; it's purely config.

### D-4.7 TFT tests run in a separate process from xgboost tests 🟡
- **What:** TFT tests carry a ``tft`` pytest marker; ``scripts/run_tests.sh``
  runs the suite as two pytest invocations (``-m "not tft"`` then ``-m tft``)
  so xgboost and torch are never loaded into the same process.
- **Why:** xgboost and torch each bundle their own OpenMP runtime (libomp).
  On macOS, whichever loads second breaks: with xgboost first, TFT training
  froze indefinitely; after a torch-first-import mitigation was tried, xgboost
  segfaulted instead (``_meta_from_numpy``). The runtimes are simply
  incompatible in one process on that platform, so load-order tricks and
  ``KMP_DUPLICATE_LIB_OK`` were abandoned in favor of process isolation,
  which is guaranteed (each half passes independently on the affected
  machine). **Scope:** tests only. Production scripts train one model per
  process and are unaffected; Linux runs both halves fine either way.
- **Change it:** Drop the marker split once upstream wheels share a single
  libomp, or if the suite moves to per-file process isolation (pytest-xdist).

---

## Phase 3 — Simulation core (Features 6, 7, 8)

### D-5.1 Unified `State` shape matching the PPO observation 🟡
- **What:** ``State`` was redesigned from a single ``forecast: ForecastOutput``
  + ``int`` day-of-week into separate ``forecast_mean`` / ``forecast_std``
  vectors, a one-hot ``day_of_week``, and a ``time_index`` — matching the PRD's
  PPO observation (Feature 7). EOQ was updated to read the new fields; the
  scaffold's ``test_state.py`` (from the initial commit) now validates this shape.
- **Why:** The environment's native per-day observation and the policies' input
  should be one object. This shape flattens almost directly into the PPO Box
  observation, so classical policies and PPO provably face the same state
  (supports the H1 comparison). Chosen over keeping the leaner ForecastOutput
  shape because only EOQ used the old one and PPO needs the split vectors.
- **Change it:** Revert ``policies/base.py`` and re-add a ``forecast`` field, but
  then PPO's wrapper must reconstruct the split vectors itself.

### D-5.2 EOQ / order-up-to safety stock reads `forecast_std`, not a scalar RMSE 🔴
- **What:** Both classical policies now source the daily forecast-error SD
  ``sigma_d`` from the state's ``forecast_std`` vector (mean over the relevant
  window) rather than a single ``historical_rmse`` scalar. EOQ:
  ``ss = z * mean(forecast_std[:L]) * sqrt(L)``. Order-up-to:
  ``ss = z * mean(forecast_std[:R+L]) * sqrt(R+L)``.
- **Why:** Keeps D-1.1's interpretation (``sigma`` is a *daily* error SD scaled
  by ``sqrt(interval)``) while letting uncertainty vary by day and using the
  exact same signal PPO sees in its observation. Supersedes the earlier
  "historical_rmse proxy" wording of D-1.1 for the in-simulation policies.
- **Change it:** Feed a flat ``forecast_std`` (every entry = ``historical_rmse``)
  to recover the old scalar behavior; the runner controls what fills the vector.

### D-7.1 One environment, two observation projections 🟡
- **What:** ``InventoryEnv`` exposes both a flat ``Box`` observation (for PPO via
  the standard Gym API) and a structured :class:`State` via ``current_state()``
  (for classical policies). Both are built from the same internal variables —
  ``_observation()`` literally flattens ``current_state()``.
- **Why:** Guarantees the two policy families see identical information, so any
  performance gap is attributable to the policy, not to differing inputs.
- **Change it:** Nothing; this is core to the experimental design.

### D-7.2 Classical policies map continuous orders to the discrete action grid 🟡
- **What:** EOQ / order-up-to compute a continuous order quantity; the runner
  converts it to the nearest discrete action via ``InventoryEnv.order_units``
  (argmin distance to the ``{0, 0.5, … 5.0} × d_bar`` grid).
- **Why:** The PRD fixes a ``Discrete(11)`` action space for PPO; running
  classical policies through the *same* action space keeps the comparison fair
  (both are limited to the same achievable order quantities). The alternative —
  letting classical policies order exact continuous amounts — would give them an
  unfair granularity advantage over PPO.
- **Change it:** Have the runner call ``env.step`` with a continuous quantity via
  a separate code path if you want classical policies unconstrained (changes the
  comparison's fairness).

### D-7.3 Arrivals processed before the day's order; init stock = first forecast 🟢
- **What:** Each ``step`` rolls the pipeline and adds arrivals *before* placing
  today's order (an order can't arrive the day it's placed), with orders landing
  in pipeline slot ``lead - 1`` so realized lead time is exactly ``lead`` days.
  ``reset`` initializes on-hand to ``forecast_mean[0]`` (a neutral non-empty
  starting stock) rather than zero.
- **Why:** Standard lost-sales inventory timing; verified day-by-day in tests.
  A non-empty start avoids a guaranteed day-1 stockout artifact.
- **Change it:** Initialize on-hand to 0 or a configured value in ``reset`` if a
  cold start is preferred.

### D-8.1 Disruptions are env wrappers; demand spike is unforecast 🟡
- **What:** Demand spike (×1.5) and lead-time disruption (×2) are
  ``gymnasium.Wrapper`` subclasses (PRD Feature 8). The spike scales the env's
  *realized* demand only — the forecast the agent sees is unchanged, so the
  disruption is a genuine surprise. Lead-time disruption inflates the env's
  mutable working lead time within the window, restored after each step.
- **Why:** Wrappers keep the core env untouched (PRD requirement). Not warning
  the agent is the point of a resilience stress test.
- **Change it:** Also perturb ``forecast_mean`` in the spike wrapper if you want
  the agent to anticipate the disruption.

---

## Phase 4 — PPO integration (Feature 9)

### D-9.1 PPO and the env share one state-flattening function 🟡
- **What:** ``simulation/environment.py::state_to_observation`` is the single
  place a :class:`State` becomes the flat observation vector. The env's
  ``_observation`` and ``PPOAgent.select_action`` both call it.
- **Why:** If the env flattened the state one way during training and the agent
  another way at decision time, PPO would see inconsistent inputs and silently
  underperform. One function guarantees they match.
- **Change it:** Nothing; this is a correctness invariant.

### D-9.2 Training episodes randomized via an episode factory 🟡
- **What:** ``InventoryEnv`` gained an optional ``episode_factory`` callable
  ``(rng) -> EpisodeData`` invoked on each ``reset``.
  ``make_training_episode_factory`` samples a random start index into the
  training series each episode (PRD "episodes randomized across the training
  period"). Evaluation passes no factory (fixed window).
- **Why:** PPO must see varied episodes to generalize rather than overfit one
  demand path. The factory keeps this opt-in so the deterministic eval/test
  path is unchanged.
- **Change it:** Pass a different factory (e.g. fixed start) to change the
  training distribution.

### D-9.3 Env generates stochastic demand from the forecast 🔴
- **What:** ``EpisodeData.demand`` is now optional. When absent, the env
  generates demand on each reset as
  ``round(max(0, forecast_mean * LogNormal(-s^2/2, s)))`` with ``s`` the per-day
  coefficient of variation ``forecast_std/forecast_mean`` (clipped to [0, 2]).
  The ``-s^2/2`` drift makes the multiplier mean 1, so demand is unbiased around
  the forecast (PRD Feature 7: "demand = forecast_value × lognormal_noise").
- **Why:** Replications differ precisely by their demand draws, so the env must
  generate demand rather than take it fixed. Using the CV as the lognormal shape
  ties demand volatility to forecast uncertainty. **This affects every
  simulation result**, so the exact noise model is worth scrutiny.
- **Change it:** Adjust the ``s`` mapping or distribution in
  ``InventoryEnv._resolve_demand``. Pass an explicit ``demand`` array to bypass
  generation entirely (deterministic replays / tests).

### D-9.4 PPO returns order units, not the raw action index 🟡
- **What:** ``PPOAgent.select_action`` returns an order quantity in units
  (chosen ``multiplier * d_bar``), matching the classical policies' contract;
  the runner maps it back to the discrete action via ``env.order_units``. The
  raw index is available separately via ``action_index``.
- **Why:** Keeps the ``Policy.select_action`` contract uniform across EOQ /
  OrderUpTo / PPO so the experiment runner treats all policies identically
  (D-7.2). The units → index round-trip is lossless because PPO's units come
  from the same grid ``order_units`` inverts.
- **Change it:** Have the runner special-case PPO and call ``action_index``
  directly if you prefer to skip the round-trip.

### D-9.5 PPO frozen-forecaster signal = sales level + RMSE uncertainty 🔴
- **What:** ``build_forecast_arrays`` sets ``forecast_mean`` to the series'
  sales level and ``forecast_std`` to the forecaster's ``historical_rmse`` held
  constant per day. The env then adds demand noise (D-9.3). The forecaster is
  loaded read-only; only its RMSE enters the simulation.
- **Why:** A pragmatic realization of "frozen forecaster generates the forecast
  in the state" that avoids rolling re-forecasts at every historical day (which
  the PRD does not require for training). The forecaster's *accuracy* (RMSE) is
  what differentiates forecasters in the simulation, which is the H3 question.
- **Change it:** Replace with true rolling per-day forecasts in
  ``simulation/episodes.py`` if you want the forecaster's bias (not just its
  variance) to flow into the sim — heavier, and a Phase 5+ consideration.

### D-9.6 ``ppo.py`` imports simulation lazily (circular-import break) 🟢
- **What:** ``PPOAgent`` imports ``state_to_observation`` / ``ACTION_MULTIPLIERS``
  inside the methods that use them, not at module top.
- **Why:** ``simulation`` imports ``policies`` (for ``State``) and ``policies``
  imports ``simulation`` (for the flattener); a top-level import created a cycle
  that silently dropped ``PPOAgent`` from the package namespace. Lazy imports
  break the cycle cleanly.
- **Change it:** Move the flattener to a neutral module both packages import, if
  a non-lazy import is preferred.

---

## Phase 5 — Experiment orchestration (Features 10, 11)

### D-10.1 The runner is pure: it takes a built env + policy 🟢
- **What:** ``simulation/runner.py::run_replications(env, policy, n, seeds,
  disruption_window)`` drives any policy through any (already-wrapped) env and
  returns an ``ExperimentResult``. Loading forecasters, building policies, and
  wrapping disruptions live in ``scripts/run_experiment.py``, not the runner.
- **Why:** Keeps the runner trivially testable with classical policies and no
  trained artifacts, and keeps artifact/config plumbing out of the hot loop.
- **Change it:** Nothing; this separation is deliberate.

### D-10.2 One Parquet per cell: daily rows + a summary row 🟢
- **What:** ``result_to_dataframe`` writes every ``(replication, day)`` row with
  ``record_type='daily'`` plus a single ``record_type='summary'`` row carrying
  the aggregate metrics (PRD Feature 10 layout). Daily and summary columns
  coexist with NaNs where not applicable.
- **Why:** Satisfies the PRD's "one row per (replication, day) plus a summary
  row" literally while keeping everything in one queryable file per cell.
- **Change it:** Split into two files (daily / summary) if a single mixed table
  is awkward for downstream analysis.

### D-10.3 Resilience metrics are within-run, not cross-condition 🔴
- **What:** ``service_level_degradation`` is the drop in mean daily service
  during the disruption window vs the pre-window period of the *same* run.
  ``recovery_time`` is days after the window until 3-day-smoothed service returns
  within 0.02 of the pre-window level (else the remaining days). Baseline runs
  (no window) report 0 for both.
- **Why:** Self-contained per run, so resilience is defined without pairing each
  disrupted run to a specific baseline run. Affects every resilience number, so
  flagged.
- **Change it:** Redefine degradation against the baseline condition's fill rate
  in ``evaluation/metrics.py`` if a cross-condition contrast is preferred.

### D-11.1 Filenames are never re-parsed for labels 🟡
- **What:** ``run_full_suite.py`` iterates ``itertools.product(forecasters,
  policies, conditions)`` and passes the labels explicitly to ``_summary_row``;
  it never splits the ``{f}_{p}_{c}.parquet`` stem back into labels.
- **Why:** Policy ``order_up_to`` and conditions ``demand_spike`` /
  ``lead_time_disruption`` contain underscores, so ``stem.split("_")`` is
  ambiguous and mislabels rows. Carrying labels from the loop is unambiguous.
- **Change it:** If a standalone re-parser is ever needed, match against the
  known policy/condition vocabularies rather than splitting on "_".

### D-11.2 Full suite is resumable via existing-file skip 🟢
- **What:** Each cell whose ``results/simulations/{f}_{p}_{c}.parquet`` already
  exists is skipped; cells run as independent subprocesses of
  ``run_experiment.py``.
- **Why:** A 27-cell × 30-rep run is long; resumability lets an interrupted run
  continue, and subprocess isolation keeps one cell's torch/xgboost state from
  leaking into the next (relevant to the libomp split, D-4.7).
- **Change it:** Delete a cell's Parquet to force a re-run.

### D-11.3 H3 correlation guards against constant input 🟢
- **What:** ``rmse_cost_correlation`` returns ``nan`` when fewer than 3 cells or
  when forecast RMSE / cost is constant (e.g. a single-forecaster subset), so it
  emits no ``ConstantInputWarning``.
- **Why:** The real 3-forecaster suite varies RMSE, but partial runs and tests
  may not; a clean ``nan`` is better than a runtime warning.
- **Change it:** Nothing.

---

_Last updated: Phase 5 (orchestration). Append new entries as later features land._
