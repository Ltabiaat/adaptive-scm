#!/usr/bin/env bash
# Run the test suite in separate processes so xgboost and torch never share one.
#
# Why: xgboost and torch (TFT + PPO) each bundle their own OpenMP runtime;
# loading both into a single process deadlocks or segfaults on macOS
# (design decision D-4.7). Splitting the run is the guaranteed fix.
#
# Usage:
#   bash scripts/run_tests.sh             # unit suite, two process groups
#   bash scripts/run_tests.sh -v          # verbose (passed through)
#   bash scripts/run_tests.sh --integration   # also run the slow integration gate
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_INTEGRATION=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--integration" ]]; then
    RUN_INTEGRATION=1
  else
    ARGS+=("$arg")
  fi
done

# Group A: everything that is NOT torch-backed (includes xgboost, no torch).
uv run pytest tests/unit/ -m "not tft and not ppo" "${ARGS[@]}"
# Group B: torch-backed unit tests (TFT + PPO), no xgboost in this process.
uv run pytest tests/unit/ -m "tft or ppo" "${ARGS[@]}"

# Optional: the slow integration gate (torch-backed; PPO 5000-step training).
if [[ "$RUN_INTEGRATION" -eq 1 ]]; then
  uv run pytest tests/integration/ "${ARGS[@]}"
fi
