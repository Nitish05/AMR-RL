"""Trajectory metrics of the evaluation scripts (scripts/eval/common.py): the driving time and
distance count only the intervals in which a forward or reverse drive was commanded (no simulator)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _common():
    spec = importlib.util.spec_from_file_location("eval_common", ROOT / "scripts" / "eval" / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driving_seconds_and_distance_count_only_commanded_drives():
    common = _common()
    rows, x = [], 0.0
    for k in range(40):
        driving = 10 <= k < 30  # 2 s of driving at 0.2 m/s between turns in place
        rows.append({"t": 0.1 * k, "gt": [x, 0.0, 0.0], "est": [x, 0.0, 0.0], "err": 0.0, "herr": 0.0,
                     "sigma": 0.01, "status": "tracking", "cmd": [0.2, 0.0] if driving else [0.0, 0.3]})
        if driving:
            x += 0.02
    m = common.trajectory_metrics(rows)
    assert m["driving_seconds"] == pytest.approx(2.0)
    assert m["driving_path_length_m"] == pytest.approx(0.4) == pytest.approx(m["true_path_length_m"])
    rows_without_cmd = [{**r, "cmd": None} for r in rows]
    assert common.trajectory_metrics(rows_without_cmd)["driving_seconds"] == 0.0
