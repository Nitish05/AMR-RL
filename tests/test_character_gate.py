"""The character gate's scorer finds learning runs where learning.py writes them (evaluation
tooling; no simulator)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    spec = importlib.util.spec_from_file_location("character_gate", ROOT / "scripts" / "eval" / "character_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase(label, **over):
    rec = {"phase": {"label": label}, "frame_valid": True, "sim_seconds": 480.0, "relocalized_at": 10.0,
           "activity_frames": {"explore": 50, "engage": 30, "idle": 20}, "decisions": [], "interactions": [],
           "outcomes": [], "contacts": [], "switches": [],
           "trajectory": {"true_path_length_m": 47.0, "lost_frames": 5, "frames": 100}}
    rec.update(over)
    return rec


def _write(run_dir, nested, label="train"):
    where = run_dir / "history_a" if nested else run_dir
    where.mkdir(parents=True)
    (where / "result.json").write_text(json.dumps({"phases": [_phase(label), _phase("test_after_restart")]}))


def test_runs_are_found_one_level_down_where_learning_py_writes_them(tmp_path):
    """The 2026-10-12 temperament batch scored as "0 runs": the results sit in
    ``T-<name>/<experiment>/result.json``, and the scorer looked only in ``T-<name>/``."""
    gate = _gate()
    _write(tmp_path / "T-default-s0", nested=True)
    _write(tmp_path / "T-caution-0.8-s1", nested=True)
    _write(tmp_path / "T-caution-0.2-s2", nested=False)  # a result directly in the run directory still counts
    (tmp_path / "notes").mkdir()
    runs = gate.load_runs(tmp_path)
    assert [r["name"] for r in runs] == ["T-caution-0.2-s2", "T-caution-0.8-s1", "T-default-s0"]
    by = {r["name"]: r for r in runs}
    assert by["T-default-s0"]["values"] == gate.DEFAULTS and by["T-default-s0"]["seed"] == 0
    assert by["T-caution-0.8-s1"]["values"]["caution"] == 0.8 and by["T-caution-0.8-s1"]["seed"] == 1
    assert all("signatures" in r for r in runs)
    assert abs(by["T-default-s0"]["signatures"]["speed_mps"] - 0.1) < 1e-9  # 47 m over 470 active seconds


def test_a_run_without_a_valid_train_phase_is_reported_as_failed(tmp_path):
    gate = _gate()
    _write(tmp_path / "T-default-s0", nested=True, label="warmup")
    runs = gate.load_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["failed"] == "no valid train phase" and "signatures" not in runs[0]
