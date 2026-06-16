#!/usr/bin/env bash
# Run ingestion for one subject or all subjects.
#
# Usage:
#   ./scripts/run_ingest.sh programming_in_data_science
#   ./scripts/run_ingest.sh all
#   ./scripts/run_ingest.sh all --skip-contextualize

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <subject_slug|all> [--skip-contextualize]"
    exit 1
fi

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

exec python -m src.ingest.run --subject "$@"
