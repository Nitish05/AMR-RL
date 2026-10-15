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


def _interactions(fixtures, t0=100.0):
    return [{"t": t0 + 20.0 * i, "started": t0 + 20.0 * i - 5.0, "action": "signal", "entity_id": f"e{i}",
             "fixture": f, "status": "outcome"} for i, f in enumerate(fixtures)]


def test_the_primary_signature_is_reported_even_where_a_level_never_defines_it(tmp_path):
    """The 2026-10-13 variance batch: at caution 0.9 the robot never interacted, so its primary
    (bloom versus stone share) was undefined in every run of that level. The scorer had dropped
    the column from the medians table and computed the ICC over the two remaining levels without
    saying so."""
    import argparse

    gate = _gate()
    for seed in range(3):
        run = tmp_path / f"T-default-s{seed}" / "style_variance"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps({"phases": [_phase("train", interactions=_interactions(
            ["bloom", "stone", "stone", "bloom"][: 3 + seed % 2]))]}))
        run = tmp_path / f"T-caution-0.1-s{seed}" / "style_variance"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps({"phases": [_phase("train", interactions=_interactions(
            ["stone", "stone", "bloom"]))]}))
        run = tmp_path / f"T-caution-0.9-s{seed}" / "style_variance"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps({"phases": [_phase("train", interactions=[])]}))
    gate.score(argparse.Namespace(dir=str(tmp_path), primary=["caution=frac_bloom_vs_stone"]))
    report = json.loads((tmp_path / "gate.json").read_text())
    caution = report["traits"]["caution"]
    assert caution["primary"] == "frac_bloom_vs_stone" and "frac_bloom_vs_stone" in caution["columns"]
    assert caution["by_level"]["0.9"]["frac_bloom_vs_stone"] is None
    assert abs(caution["by_level"]["0.1"]["frac_bloom_vs_stone"] - 1 / 3) < 1e-9
    assert caution["icc_levels"] == 2 and caution["levels"] == 3 and caution["pass"] is False
    md = (tmp_path / "gate.md").read_text()
    assert "(2 of 3 levels have the primary)" in md
    header = next(line for line in md.splitlines() if line.startswith("| level | n |"))
    assert "frac_bloom_vs_stone" in header


def test_second_prereg_signatures_driving_speed_and_attempts_on_the_object_made_inert(tmp_path):
    """2026-10-15: approach tempo is scored on the speed while a drive is commanded, surprise
    weighting on the attempts aimed at the object the switch made inert, and caution reports the
    share of revisits to an entity whose last outcome was aversive."""
    import pytest

    gate = _gate()
    inter = [{"t": 100.0, "started": 90.0, "action": "signal", "entity_id": "b", "fixture": "bloom", "status": "outcome"},
             {"t": 160.0, "started": 150.0, "action": "signal", "entity_id": "g", "fixture": "grump", "status": "outcome"},
             {"t": 200.0, "started": 190.0, "action": "signal", "entity_id": "g", "fixture": "grump", "status": "aborted"},
             {"t": 320.0, "started": 310.0, "action": "signal", "entity_id": "b", "fixture": "bloom", "status": "outcome"},
             {"t": 350.0, "started": 340.0, "action": "signal", "entity_id": "b", "fixture": "bloom", "status": "aborted"},
             {"t": 400.0, "started": 390.0, "action": "signal", "entity_id": "s", "fixture": "stone", "status": "outcome"}]
    outs = [{"t": 100.0, "entity_id": "b", "fixture": "bloom", "valence": 1.0},
            {"t": 160.0, "entity_id": "g", "fixture": "grump", "valence": -1.0},
            {"t": 320.0, "entity_id": "b", "fixture": "bloom", "valence": 0.0},
            {"t": 400.0, "entity_id": "s", "fixture": "stone", "valence": 1.0}]
    run = tmp_path / "T-surprise_weighting-0.9-s3" / "style_reversal"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"phases": [_phase(
        "train", sim_seconds=660.0, interactions=inter, outcomes=outs, switches=[{"t": 300.0, "to": "swapped"}],
        trajectory={"true_path_length_m": 47.0, "lost_frames": 5, "frames": 100, "driving_seconds": 200.0,
                    "driving_path_length_m": 40.0})]}))
    row = gate.load_runs(tmp_path)[0]
    assert row["seed"] == 3 and row["values"]["surprise_weighting"] == 0.9 and row["idle"] is False
    sig = row["signatures"]
    assert sig["drive_speed_mps"] == pytest.approx(0.2)
    assert sig["post_switch_inert_attempts"] == 2 and sig["post_switch_inert_last_s"] == pytest.approx(50.0)
    # revisits with an earlier outcome on the same entity: grump (last outcome red), bloom x2 (last +1, then 0)
    assert sig["aversive_revisit_frac"] == pytest.approx(1 / 3)
    assert gate.PRIMARY["approach_tempo"] == "drive_speed_mps"
    assert gate.PRIMARY["surprise_weighting"] == "post_switch_inert_attempts"
    assert gate.PRIMARY["caution"] == "engage_frac" and gate.TEMPERAMENT == ("approach_tempo", "caution", "novelty_bias")
