#!/usr/bin/env bash
# AMR-RL task runner. Uses $AMR_PYTHON, else .venv/bin/python, else python3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${AMR_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)
    # Isolated environment for AMR-RL only (does not touch BB8-RL or Studio envs).
    python3.12 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/python" -m pip install --upgrade pip
    "$ROOT/.venv/bin/python" -m pip install -e ".[simulation,dev]"
    ;;
  test)
    "$PY" -m pytest -q -p no:cacheprovider -m "not genesis" "$@"
    "$PY" -m ruff check --no-cache src tests scripts
    node --check src/amr_rl/ui/static/app.js
    node tests/test_ui_static.js
    ;;
  test-sim)
    RUN_GENESIS=1 "$PY" -m pytest -q -p no:cacheprovider -m genesis "$@"
    ;;
  generate-robot)
    "$PY" -m amr_rl.robot.generate "$@"
    ;;
  app)
    "$PY" -m amr_rl.app "$@"
    ;;
  map)
    "$PY" scripts/eval/build_map.py "$@"
    ;;
  eval-nav)
    "$PY" scripts/eval/navigation.py "$@"
    ;;
  eval-learning)
    "$PY" scripts/eval/learning.py "$@"
    ;;
  *)
    echo "usage: scripts/amr.sh {setup|test|test-sim|generate-robot|app|map|eval-nav|eval-learning} [args]"
    ;;
esac
