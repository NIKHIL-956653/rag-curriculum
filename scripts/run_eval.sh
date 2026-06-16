#!/usr/bin/env bash
# Run the evaluation harness against the eval set.
#
# Usage:
#   ./scripts/run_eval.sh
#   ./scripts/run_eval.sh --output-json results.json

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

exec python -m src.eval.run_eval "$@"
