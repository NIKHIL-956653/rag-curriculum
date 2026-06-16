#!/usr/bin/env bash
# One-shot project setup. Run from project root.
#
# Steps:
#   1. Create Python venv and install deps
#   2. Start Docker infrastructure
#   3. Pull Ollama models
#   4. Verify everything is reachable
#
# Usage:
#   ./scripts/setup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Detect activate script (Linux/macOS vs Windows Git Bash)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "Could not find venv activate script" >&2
    exit 1
fi

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Copying .env if missing..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "    Created .env from .env.example"
fi

echo "==> Starting Docker services..."
docker compose up -d

echo "==> Waiting for services to be ready..."
sleep 6

echo "==> Pulling Ollama models (if not already present)..."
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

echo "==> Verifying connectivity..."
python -c "
from src.utils.db import get_qdrant, pg_cursor, ensure_collection
from src.utils.ollama_client import get_ollama

print('Qdrant... ', end='')
c = get_qdrant()
collections = [x.name for x in c.get_collections().collections]
print('OK')

print('Postgres... ', end='')
with pg_cursor() as cur:
    cur.execute('SELECT 1')
    cur.fetchone()
print('OK')

print('Ollama... ', end='')
o = get_ollama()
v = o.embed('hello')
print(f'OK (embedding dim={len(v)})')

print('Creating Qdrant collection (if needed)...')
ensure_collection()
print('All systems go.')
"

echo ""
echo "==> Setup complete. Next steps:"
echo "    1. Drop PDFs into data/raw/<subject_slug>/ (slugs in configs/subjects.yaml)"
echo "    2. Ingest the pilot subject:"
echo "       ./scripts/run_ingest.sh programming_in_data_science"
echo "    3. Launch the UI:"
echo "       streamlit run app/streamlit_app.py"
