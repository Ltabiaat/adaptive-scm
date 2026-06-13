#!/usr/bin/env bash
# Run the unit suite in two processes so xgboost and torch never share one.
#
# Why: xgboost and torch each bundle their own OpenMP runtime; loading both
# into a single process deadlocks or segfaults on macOS (design decision
# D-4.7). Splitting the run is the guaranteed fix. Extra pytest args are
# passed through to both invocations, e.g.:
#
#   bash scripts/run_tests.sh            # full suite (76 tests)
#   bash scripts/run_tests.sh -v         # verbose
set -euo pipefail
cd "$(dirname "$0")/.."

uv run pytest tests/unit/ -m "not tft" "$@"
uv run pytest tests/unit/ -m tft "$@"
