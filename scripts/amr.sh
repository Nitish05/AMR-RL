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
    "$ROOT/.venv/bin/python" -m pip install -e ".[simulation,dev,depth,place]"
    ;;
  bench-setup)
    # Light environment for the learner testbed only (no Genesis, no torch): .venv-bench.
    # Uses uv if installed (it fetches Python 3.12 itself), else a Homebrew/system Python >= 3.10.
    VENV="$ROOT/.venv-bench"
    PKGS="numpy scipy opencv-python-headless pyyaml pytest ruff"
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 "$VENV"
      uv pip install --python "$VENV/bin/python" $PKGS
    else
      for cand in python3.12 python3.13 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
          "$cand" -m venv "$VENV"; break
        fi
      done
      if [[ ! -x "$VENV/bin/python" ]]; then
        echo "Need Python >= 3.10. Install uv (https://docs.astral.sh/uv/) or: brew install python@3.12" >&2; exit 1
      fi
      "$VENV/bin/python" -m pip install --upgrade pip
      "$VENV/bin/python" -m pip install $PKGS
    fi
    "$VENV/bin/python" -c "import numpy, cv2, yaml; print('bench environment ready:', numpy.__version__, cv2.__version__)"
    ;;
  bench-test)
    "$ROOT/.venv-bench/bin/python" -m pytest -q -p no:cacheprovider -m "not genesis" \
      tests/test_learner_bench.py tests/test_learning_memory.py tests/test_perception_units.py "$@"
    ;;
  learner-bench)
    "$ROOT/.venv-bench/bin/python" scripts/eval/learner_bench.py "$@"
    ;;
  fetch-place-model)
    # Downloads the pinned MegaLoc place-recognition code (GitHub commit) and weights
    # (MIT, ~900 MB) into the torch-hub and Hugging Face caches outside the repository.
    # Without them the VSLAM runs without loop closure and reports so in its state.
    "$PY" -c "from amr_rl.perception.place_recognition import MegaLocDescriptor; MegaLocDescriptor(allow_download=True); print('place model ready')"
    ;;
  fetch-depth-model)
    # Downloads the pinned Depth Anything V2 Small weights (Apache-2.0, ~100 MB) into the
    # Hugging Face cache outside the repository. Without them the near-field guard is off.
    "$PY" -c "from amr_rl.perception.near_depth import load_backend; import sys; sys.exit(0 if load_backend(allow_download=True) else 1)"
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
    echo "usage: scripts/amr.sh {setup|bench-setup|bench-test|learner-bench|fetch-depth-model|fetch-place-model|test|test-sim|generate-robot|app|map|eval-nav|eval-learning} [args]"
    ;;
esac
