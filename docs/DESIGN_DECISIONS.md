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

_Last updated: Phase 3 (XGBoost). Append new entries as later features land._
