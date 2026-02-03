#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create it with: python3.11 -m venv .venv" >&2
  exit 1
fi

# macOS: ensure .pth files are processed (site-packages must not be hidden)
if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ -d ".venv/lib/python3.11/site-packages" ]]; then
    chflags -R nohidden ".venv/lib/python3.11/site-packages" 2>/dev/null || true
  fi
fi

source ".venv/bin/activate"

CONFIG="${1:-config/config.json}"

python -m spine.main --config "$CONFIG" --dry-run --debug-view
