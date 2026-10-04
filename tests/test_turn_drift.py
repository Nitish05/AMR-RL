"""Heading drift in in-place turns (round 7): the turn-handling switches on a synthetic
scene (tests/turn_scene.py; no simulator). Benchmark and live results:
docs/results/turn-drift.md."""

import json
import math
from pathlib import Path

import numpy as np
from turn_scene import run_turn

from amr_rl.perception.vslam import PlanarVSLAM, VSLAMConfig

BASELINE = Path(__file__).parent / "data" / "turn_baseline_poses.json"


def final_error(rows):
    return [r[2] for r in rows if r[2] is not None][-1]


def test_switches_off_leave_tracking_unchanged():
    """Recorded with the code before any turn switch existed."""
    _, rows = run_turn(box_dist=0.30, turn_deg=360, start_heading=math.pi)
    now = [None if r[4].pose is None else [round(float(v), 9) for v in r[4].pose] for r in rows]
    assert now == json.loads(BASELINE.read_text())["box030_360"]


def test_turn_state_follows_the_command():
    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg, slam.pose, slam._turn, slam._turn_count = VSLAMConfig(), np.zeros(3), None, 0
    slam._update_turn_state((0.0, 0.45), 0.1, 0.1)
    assert slam._turn is None  # one frame is not a turn
    slam._update_turn_state((0.0, 0.45), 0.1, 0.2)
    assert slam._turn is not None
    slam._update_turn_state((0.0, 0.0), 0.1, 0.6)  # a short pause keeps the turn
    assert slam._turn is not None
    slam._update_turn_state((0.15, 0.0), 0.1, 0.7)  # driving ends it
    assert slam._turn is None


def test_heading_only_turns_pin_the_axis_and_fall_back_when_the_base_moves():
    cfg = VSLAMConfig(turn_pin_xy="hard")
    slam, rows = run_turn(cfg, box_dist=None, turn_deg=120)
    turning = [r[4] for r in rows if r[4].info.get("turn")]
    assert turning and max(np.hypot(*(r.pose[:2] - turning[0].pose[:2])) for r in turning) < 1e-9
    slam, rows = run_turn(cfg, box_dist=None, turn_deg=120, base_drift=(0.3, 0.0))
    assert slam._turn is None or slam._turn["fallbacks"] > 0
    assert max(r[3] for r in rows if r[3] is not None) < 0.05  # it followed the real 0.3 m/s slide


def test_turn_closure_removes_the_heading_error_a_near_box_causes():
    """Box face 0.30 m from the axis: the floor-lifted face points cost ~12 deg per turn."""
    _, rows = run_turn(VSLAMConfig(), box_dist=0.30, turn_deg=360, start_heading=math.pi)
    assert abs(final_error(rows)) > 8
    slam, rows = run_turn(VSLAMConfig(turn_closure="rotate"), box_dist=0.30, turn_deg=360, start_heading=math.pi)
    assert any(c.get("outcome") == "closed" for c in slam.turn_closures)
    assert abs(final_error(rows)) < 1.0


def test_turn_closure_does_not_move_a_correct_turn():
    slam, rows = run_turn(VSLAMConfig(turn_closure="rotate"), box_dist=None, turn_deg=360)
    assert abs(final_error(rows)) < 0.5
    assert all(abs(c.get("delta_deg", 0)) < 1.0 for c in slam.turn_closures if c.get("outcome") == "closed")


def test_floor_validation_confirms_floor_points_by_parallax():
    slam, _ = run_turn(VSLAMConfig(floor_validation="all"), box_dist=None, turn_deg=180)
    lm = slam.lm
    validated = lm.alive & ((lm.flags & 2) != 0)
    assert (validated & (lm.kind == 0)).sum() > 100
    # on an empty floor almost nothing may be mistaken for an object
    assert (validated & (lm.kind == 1)).sum() < 0.1 * validated.sum()


def test_commanded_turn_check_flags_over_rotation_and_freezes_only():
    slam = PlanarVSLAM.__new__(PlanarVSLAM)
    slam.cfg = VSLAMConfig(turn_cmd_check="flag")
    for cmd, est, flagged in [(math.pi, 0.83 * math.pi, False), (math.pi, 0.7 * math.pi, False),
                              (math.pi, 1.25 * math.pi, True), (math.pi, 0.1, True), (0.3, 0.0, False)]:
        tr = {"cmd_rot": cmd, "est_rot": est, "flagged": False}
        slam._turn_command_check(tr)
        assert tr["flagged"] == flagged, (cmd, est)


def test_alignment_turn_continues_on_turn_prediction_only_when_enabled():
    from types import SimpleNamespace

    from amr_rl.behavior.activities import Goto

    class Nav:
        def __init__(self, cmd):
            self.cmd, self.status, self.reason = cmd, "following", ""

        def set_goal(self, *a, **k):
            return True

        def step(self, *a):
            return self.cmd

    def rt(cmd, enabled, turning=True):
        return SimpleNamespace(loc_status="predicted", pose=np.zeros(3), sigma=0.01, grid=None, nav=Nav(cmd),
                               cfg=SimpleNamespace(respect_degraded_heading=enabled),
                               slam=SimpleNamespace(_turn={} if turning else None), avoid_regions=lambda: [])

    g = Goto((1.0, 0.0))
    assert g.step(rt((0.0, 0.4), enabled=False), 0.0) == (0.0, 0.0)
    g = Goto((1.0, 0.0))
    assert g.step(rt((0.0, 0.4), enabled=True), 0.0) == (0.0, 0.4)  # pure rotation continues
    g = Goto((1.0, 0.0))
    assert g.step(rt((0.2, 0.1), enabled=True), 0.0) == (0.0, 0.0)  # never drive on a prediction
    g = Goto((1.0, 0.0))
    assert g.step(rt((0.0, 0.4), enabled=True, turning=False), 0.0) == (0.0, 0.0)


def test_turn_preset_sets_the_finalist_and_degraded_heading_handling():
    from amr_rl.perception.vslam import TURN_PRESETS
    from amr_rl.runtime.robot import RuntimeConfig, apply_turn_preset

    cfg = apply_turn_preset(RuntimeConfig(), "fam")
    assert cfg.respect_degraded_heading
    assert all(getattr(cfg.vslam, k) == v for k, v in TURN_PRESETS["fam"].items())
    assert not apply_turn_preset(RuntimeConfig(), "none").respect_degraded_heading
