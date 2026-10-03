"""Pure perception/identity/outcome contracts (no simulator)."""

import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from amr_rl.learning.identity import EntityTracker
from amr_rl.learning.memory import ExperienceMemory
from amr_rl.learning.outcomes import classify, displacement_threshold
from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.entities import Detection, FixtureDetector, hue_name
from amr_rl.perception.semantic import SemanticWorker
from amr_rl.robot.spec import RobotSpec


@pytest.fixture(scope="module")
def model():
    return CameraModel.from_spec(RobotSpec.load())


def test_camera_geometry_round_trip(model):
    pose = (0.3, -0.2, 0.7)
    P = np.array([[1.2, 0.1, 0.0], [0.9, -0.4, 0.0], [2.0, 0.5, 0.0]])
    uv, z = model.project_world(P, pose)
    assert (z > 0).all()
    back, ok = model.ground_points_world(uv, pose, max_range=5)
    assert ok.all() and np.allclose(back, P, atol=1e-6)
    H = model.floor_homography(pose)
    h = (H @ np.c_[P[:, :2], np.ones(3)].T).T
    assert np.allclose(h[:, :2] / h[:, 2:], uv, atol=1e-6)
    assert abs(model.camera_height - RobotSpec.load().camera["height_above_ground"]) < 1e-9


def test_scale_prior_error_scales_ground_distances(model):
    """Documented calibration sensitivity: a height error scales IPM ranges."""
    spec = RobotSpec.load()
    biased = CameraModel.from_spec(spec, height_error=0.0155)  # +10 %
    uv, _ = model.project_world(np.array([[1.5, 0.0, 0.0]]), (0, 0, 0))
    P, _ = biased.ground_points_world(uv, (0, 0, 0))
    cam = biased.T_world_cam((0, 0, 0))[:3, 3]
    ratio = (P[0, 0] - cam[0]) / (1.5 - model.T_world_cam((0, 0, 0))[0, 3])
    assert ratio == pytest.approx(1.1, rel=0.02)


def _synthetic(color_rgb, panel=None, size=(320, 240)):
    img = np.full((size[1], size[0], 3), 128, np.uint8)
    img[140:200, 140:180] = color_rgb
    if panel is not None:
        img[112:139, 145:175] = panel
    return img


def test_detector_finds_body_and_attachment(model):
    det = FixtureDetector(model)
    plain = det.detect(_synthetic((13, 199, 209)), pose=(0, 0, 0))
    assert len(plain) == 1 and plain[0].color == "cyan" and plain[0].state_token == "attach:none"
    raised = det.detect(_synthetic((13, 199, 209), panel=(250, 214, 13)), pose=(0, 0, 0))
    assert len(raised) == 1 and raised[0].state_token == "attach:yellow"
    assert raised[0].position is not None and raised[0].range_m > 0


def test_attachment_partly_occluded_by_rounded_top_edge(model):
    """Regression: close range, the body's rounded top corners rise above the
    panel's lowest row, so the bounding boxes overlap (seen in a real run: learning history_a, grump)."""
    img = _synthetic((22, 160, 60))  # green body rows 140..199, cols 140..179
    img[130:140, 140:150] = (22, 160, 60)  # corners rise above the panel's bottom row
    img[130:140, 170:180] = (22, 160, 60)
    img[100:139, 150:170] = (220, 25, 25)  # red panel resting on the body
    dets = det_for(model).detect(img)
    assert len(dets) == 1 and dets[0].state_token == "attach:red"
    # a separate blob floating well above is not an attachment
    img2 = _synthetic((22, 160, 60))
    img2[20:60, 150:170] = (220, 25, 25)
    tokens = sorted(d.state_token for d in det_for(model).detect(img2))
    assert tokens == ["attach:none", "attach:none"]


def det_for(model):
    return FixtureDetector(model)


def test_desaturated_scene_has_no_detections(model):
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320, 3)) * 60 + 100).astype(np.uint8)
    assert FixtureDetector(model).detect(img, pose=(0, 0, 0)) == []


def test_hue_names():
    assert [hue_name(h) for h in (2, 55, 130, 185, 226, 320)] == ["red", "yellow", "green", "cyan", "blue", "magenta"]


# ----------------------------------------------------------------- outcomes
def obs(token, xy, n=3, t0=0.0):
    return [{"token": token, "position": list(xy), "frame_sha256": "0" * 64, "t": t0 + 0.1 * i} for i in range(n)]


def test_outcome_classification():
    before = obs("attach:none", (1, 1))
    assert classify(before, obs("attach:yellow", (1, 1)))[0] == "attach:yellow"
    assert classify(before, obs("attach:none", (1, 1)))[0] == "none"
    assert classify(before, obs("attach:none", (1.4, 1)))[0] == "moved"
    # transient response still counts
    transient = obs("attach:red", (1, 1), 3) + obs("attach:none", (1, 1), 4)
    assert classify(before, transient)[0] == "attach:red"
    # insufficient or unstable evidence -> nothing learned
    assert classify(before[:2], obs("attach:yellow", (1, 1)))[0] is None
    assert classify(before, [])[0] is None
    assert classify(before, [], place_in_view_after=3)[0] == "moved"
    flicker = [dict(o, token=t) for o, t in zip(obs("x", (1, 1), 6), ["attach:none", "attach:blue"] * 3)]
    assert classify(before, flicker)[0] is None


def test_moved_requires_displacement_beyond_measured_position_scatter():
    rng = np.random.default_rng(0)

    def noisy(xy, n, sigma, t0=0.0):
        return [{"token": "attach:none", "position": list(np.asarray(xy) + rng.normal(0, sigma, 2)),
                 "frame_sha256": "0" * 64, "t": t0 + 0.2 * i} for i in range(n)]

    # A stationary object seen with 10 cm position noise: never "moved".
    false_moves = sum(classify(noisy((1, 1), 3, 0.10), noisy((1, 1), 12, 0.10, 5))[0] == "moved"
                      for _ in range(200))
    assert false_moves <= 6  # ~30 % with the bare 0.15 m threshold
    # The same 0.18 m shift is a move when positions are precise, not when they scatter.
    assert classify(obs("attach:none", (1, 1)), obs("attach:none", (1.18, 1)))[0] == "moved"
    before = np.array([o["position"] for o in noisy((1, 1), 3, 0.10)])
    after = np.array([o["position"] for o in noisy((1.18, 1), 12, 0.10, 5)])
    assert displacement_threshold(before, after) > 0.18
    # A large move is still seen through the noise.
    assert classify(noisy((1, 1), 6, 0.10), noisy((1.6, 1), 12, 0.10, 5))[0] == "moved"


def test_context_already_raised_is_not_a_new_response():
    before = obs("attach:yellow", (1, 1))
    assert classify(before, obs("attach:yellow", (1, 1)))[0] == "none"


# ----------------------------------------------------------------- identity
def detection(hue, xy, color="blue", width=0.2):
    return Detection(0, (0, 0, 10, 10), 100, hue, color, 0.8, 0.95, (5, 10), False, [], tuple(xy), 1.0, width, 0.2)


def test_identity_promotion_reassociation_and_twin_ambiguity(tmp_path):
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    tracker = EntityTracker(memory)
    pose = (0, 0, 0)
    for k in range(3):
        out = tracker.update([detection(226, (1, 0))], pose, 0.01, "map-1", float(k))
    assert out[0].entity_id is not None
    first = out[0].entity_id
    out = tracker.update([detection(226, (1.05, 0.02))], pose, 0.01, "map-1", 4.0)
    assert out[0].entity_id == first and not out[0].ambiguous
    # a look-alike far away, seen alone, is ambiguous (it could be the same object moved)
    alone = tracker.update([detection(226, (3, 2))], pose, 0.01, "map-1", 4.5)
    assert alone[0].ambiguous
    # seen together with the original in its place, it becomes its own entity
    for k in range(3):
        out = tracker.update([detection(226, (1, 0)), detection(226, (3, 2))], pose, 0.01, "map-1", 5.0 + k)
    twin = out[1].entity_id
    assert twin not in (None, first)
    # after a map change both lose positions: identical appearance -> ambiguous
    memory.invalidate_positions("map-2")
    tracker.refresh()
    out = tracker.update([detection(226, (2, 1))], pose, 0.01, "map-2", 9.0)
    assert out[0].ambiguous and out[0].entity_id is None


def test_identity_relocation_requires_seeing_old_place_empty(tmp_path):
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    tracker = EntityTracker(memory)
    for k in range(3):
        out = tracker.update([detection(130, (1, 0), "green")], (0, 0, 0), 0.01, "m", float(k))
    eid = out[0].entity_id
    moved = detection(130, (2.5, 1), "green")
    out = tracker.update([moved], (0, 0, 0), 0.01, "m", 5.0, in_view=lambda xy: False)
    assert out[0].ambiguous
    out = tracker.update([moved], (0, 0, 0), 0.01, "m", 6.0, in_view=lambda xy: True)
    assert out[0].entity_id == eid and out[0].confidence < 0.8


def test_duplicate_claims_become_ambiguous(tmp_path):
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    tracker = EntityTracker(memory)
    for k in range(3):
        tracker.update([detection(185, (1, 0), "cyan")], (0, 0, 0), 0.01, "m", float(k))
    out = tracker.update([detection(185, (1, 0), "cyan"), detection(185, (1.1, 0.05), "cyan")],
                         (0, 0, 0), 0.01, "m", 4.0)
    assert all(o.entity_id is None for o in out)



def _receipt(eid, observed, event, t):
    return {"event_id": event, "entity_id": eid, "action": "signal", "context": "attach:none", "observed": observed,
            "timestamp": t, "authorised": True, "images_retained": True, "predicted": {}, "decision": {}}


def test_duplicate_identity_of_one_object_is_merged_with_its_history(tmp_path):
    """Regression (learning-20260930-022340): grump was stored twice (a bad-angle view
    made the box look 10 cm wider; later refined to 0.66 m apart); the duplicate had no
    red-panel history and was re-targeted. Same appearance, close, never two separate
    detections in one frame -> merged into the older identity."""
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    tracker = EntityTracker(memory)
    for k in range(3):
        out = tracker.update([detection(130, (1.0, 0.0), "green")], (0, 0, 0), 0.01, "m", float(k))
    old = out[0].entity_id
    memory.record_outcome(_receipt(old, "attach:red", "e1", 3.0))
    for k in range(3):  # bad-angle view: box looks 13 cm wider (outside the appearance gate), 1.05 m away
        out = tracker.update([detection(130, (2.0, 0.3), "green", width=0.33)], (0, 0, 0), 0.01, "m", 20.0 + k)
    dup = out[0].entity_id
    assert dup not in (None, old) and memory.counts()["entities"] == 2
    # its position is later refined to 0.66 m from the original (as in the real run)
    memory.upsert_entity(dup, appearance=memory.entities()[1]["appearance"], now=30.0, map_version="m",
                         xy=(1.6, 0.25))
    tracker.refresh()
    tracker.update([], (0, 0, 0), 0.01, "m", 31.0)
    assert memory.counts()["entities"] == 1 and memory.counts()["merges"] == 1
    assert memory.resolve(dup) == old and tracker.merge_log[0]["kept"] == old
    assert memory.attitude(old, 32.0)["attitude"] == "disliked"  # the red-panel history is kept
    # a receipt for the merged-away id (interaction in flight) lands on the kept identity
    memory.record_outcome(_receipt(dup, "attach:red", "e2", 33.0))
    assert memory.counts()["outcomes"] == 2 and memory.counts()["entities"] == 1


def test_twins_seen_together_are_never_merged(tmp_path):
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    tracker = EntityTracker(memory)
    for k in range(3):
        tracker.update([detection(226, (1.0, 0.0))], (0, 0, 0), 0.01, "m", float(k))
    for k in range(4):  # a look-alike seen together with the original: its own identity
        out = tracker.update([detection(226, (1.0, 0.0)), detection(226, (2.2, 0.3))], (0, 0, 0), 0.01, "m",
                             5.0 + k)
    a, b = out[0].entity_id, out[1].entity_id
    assert None not in (a, b) and a != b and memory.are_distinct(a, b)
    # even if their remembered positions end up close, co-visibility forbids merging
    ent_b = [e for e in memory.entities() if e["entity_id"] == b][0]
    memory.upsert_entity(b, appearance=ent_b["appearance"], now=20.0, map_version="m", xy=(1.5, 0.2))
    tracker.refresh()
    tracker.update([], (0, 0, 0), 0.01, "m", 21.0)
    assert memory.counts()["entities"] == 2 and memory.counts()["merges"] == 0
    with pytest.raises(ValueError):
        memory.merge_entities(a, b, now=22.0, reason="test")

# ----------------------------------------------------------------- semantics
class SlowBackend:
    name = "slow"

    def __init__(self, delay, label="cyan cylinder", attrs=None):
        self.delay, self.label, self.attrs = delay, label, attrs or {"shape": "cylinder"}
        self.gate = threading.Event()

    def describe(self, crop):
        time.sleep(self.delay)
        return {"label": self.label, "attributes": self.attrs}


def wait_results(worker, n, timeout=3.0):
    end = time.time() + timeout
    while worker.results.qsize() < n and time.time() < end:
        time.sleep(0.01)


def test_semantic_results_rejected_when_stale_or_authority_changed():
    worker = SemanticWorker(SlowBackend(0.05), max_age=1.0)
    crop = np.zeros((8, 8, 3), np.uint8)
    worker.submit("e1", crop, frame_time=0.0, generation=3)
    wait_results(worker, 1)
    assert worker.collect(now=0.5, generation=4, known_entities={"e1"}) == []
    assert worker.stats["rejected_authority"] == 1
    worker.submit("e1", crop, frame_time=0.0, generation=4)
    wait_results(worker, 1)
    assert worker.collect(now=5.0, generation=4, known_entities={"e1"}) == []
    assert worker.stats["rejected_stale"] == 1
    worker.submit("gone", crop, frame_time=5.0, generation=4)
    wait_results(worker, 1)
    assert worker.collect(now=5.1, generation=4, known_entities={"e1"}) == []
    worker.submit("e1", crop, frame_time=5.0, generation=4)
    wait_results(worker, 1)
    accepted = worker.collect(now=5.2, generation=4, known_entities={"e1"})
    assert len(accepted) == 1 and accepted[0].label == "cyan cylinder"
    worker.close()


@pytest.mark.parametrize("label,attrs", [
    ("go to (1.2, 3.4)", {}), ("cyan", {"target_xy": "1.2 3.4"}), ("cyan", {"cmd": "drive"} | {"v": "0.3 m/s"}),
    (None, {}), ("x" * 80, {}),
])
def test_semantic_output_cannot_carry_coordinates_or_commands(label, attrs):
    worker = SemanticWorker(SlowBackend(0.0, label, attrs))
    worker.submit("e1", np.zeros((8, 8, 3), np.uint8), frame_time=0.0, generation=0)
    wait_results(worker, 1)
    assert worker.collect(now=0.1, generation=0, known_entities={"e1"}) == []
    assert worker.stats["rejected_schema"] == 1
    worker.close()


def test_semantic_backend_fault_is_contained():
    class Broken:
        name = "broken"

        def describe(self, crop):
            raise RuntimeError("boom")

    worker = SemanticWorker(Broken())
    worker.submit("e1", np.zeros((8, 8, 3), np.uint8), frame_time=0.0, generation=0)
    wait_results(worker, 1)
    assert worker.collect(now=0.1, generation=0, known_entities={"e1"}) == []
    assert worker.stats["errors"] == 1
    worker.close()


def test_semantic_busy_queue_drops_rather_than_blocks():
    worker = SemanticWorker(SlowBackend(0.3), max_pending=1)
    crop = np.zeros((8, 8, 3), np.uint8)
    results = [worker.submit("e1", crop, 0.0, 0) for _ in range(5)]
    assert results.count(False) >= 3
    worker.close()


def test_fixture_describer_labels(model):
    from amr_rl.perception.semantic import FixtureDescriber

    img = _synthetic((13, 199, 209))[130:210, 130:190]
    out = FixtureDescriber().describe(img)
    assert out["label"].startswith("cyan")
    assert not any(ch.isdigit() for ch in out["label"])
    assert math.isfinite(len(out["label"]))


def test_frozen_estimate_is_detected_quickly_and_poisoned_landmarks_removed(model):
    """Regression (map-arena-20260930-075954, t=323 s): vision reported the robot
    standing still while it drove 0.25 m; the long consistency window needed 0.3 m of
    commanded motion. Relocalisation then locked back onto landmarks created from the
    frozen pose (error 6 -> 33 cm)."""
    from amr_rl.perception.vslam import PlanarVSLAM

    slam = PlanarVSLAM(model)
    slam.pose = np.array([1.0, 0.0, 0.0])
    slam.lm.add(np.array([[2.0, 0.0, 0.0]]), np.zeros((1, 32), np.uint8), 0, created=5.0)   # before
    slam.lm.add(np.array([[2.0, 0.5, 0.0]]), np.zeros((1, 32), np.uint8), 0, created=10.8)  # during
    ok = True
    for k in range(13):  # 1.2 s at 0.2 m/s commanded, vision says: not moving
        ok = slam._motion_consistent((1.0, 0.0, 0.0), (0.2, 0.0), 0.1, 10.0 + 0.1 * k)
        if not ok:
            break
    assert not ok and slam.inconsistency == "frozen" and 10.0 + 0.1 * k <= 11.2 + 1e-9
    slam._on_frozen(10.0 + 0.1 * k)
    assert slam.lm.alive.tolist() == [True, False]
    assert slam._dr["pose"][0] > 1.15  # dead reckoning continued the commanded motion
    # relocalisation gate: a candidate at the frozen pose is refused, one near DR accepted
    gate = 0.12 + 0.3 * slam._dr["travel"]
    assert np.hypot(*(np.array([1.0, 0.0]) - slam._dr["pose"][:2])) > gate


def test_real_slow_start_is_not_flagged_as_frozen(model):
    from amr_rl.perception.vslam import PlanarVSLAM

    slam = PlanarVSLAM(model)
    x = 0.0
    for k in range(20):  # commanded 0.2 m/s; real motion lags then reaches 70 % of it
        x += 0.02 * min(0.7, 0.1 * k)
        assert slam._motion_consistent((x, 0.0, 0.0), (0.2, 0.0), 0.1, 0.1 * k)


def test_describer_does_not_guess_box_or_cylinder_from_a_silhouette():
    """Regression: the fill heuristic labelled the cyan cylinder 'block' and the green
    box 'cylinder'. Only a round outline is named (ball); otherwise 'object'."""
    import cv2

    from amr_rl.perception.semantic import FixtureDescriber

    d = FixtureDescriber()
    rect = np.full((60, 60, 3), 128, np.uint8)
    rect[10:50, 15:45] = (13, 199, 209)
    assert d.describe(rect)["label"] == "cyan object"
    disc = np.full((60, 60, 3), 128, np.uint8)
    cv2.circle(disc, (30, 30), 20, (220, 40, 160), -1)
    assert d.describe(disc)["attributes"]["shape"] == "ball"


def test_any_tracking_loss_gates_relocalisation_by_dead_reckoning_including_heading(model):
    """Regression (map-arena-20260930-083710, t=258 s): after an ordinary tracking loss
    relocalisation accepted a pose 19 cm / 10 deg off, and the rest of the map was
    built in that rotated frame."""
    from amr_rl.perception.vslam import LOST, PlanarVSLAM

    slam = PlanarVSLAM(model)
    slam.pose = np.array([0.0, 0.0, 0.0])
    for k in range(20):  # tracked 2 s of a slow left turn while driving
        slam.pose = np.array([0.02 * k, 0.0, 0.01 * k])
        slam._motion_consistent(slam.pose, (0.2, 0.1), 0.1, 0.1 * k)
    slam._arm_dead_reckoning(2.0, remove_landmarks=False)
    slam.status = LOST
    dr = slam._dr["pose"].copy()
    gate_h = 0.12 + 0.25 * slam._dr["rot"]
    good = dr.copy()
    bad = dr + np.array([0.0, 0.0, gate_h + 0.05])  # right place, wrong heading
    calls = iter([(bad, 60, np.eye(3)), (good, 60, np.eye(3)), (good, 60, np.eye(3))])
    slam.global_localize = lambda pts, desc: next(calls)
    r = slam._relocalize(None, None, None, 3.0)
    assert r.status != "tracking" and r.reason == "relocalization_disagrees_with_dead_reckoning"
    slam._relocalize(None, None, None, 3.1)
    r = slam._relocalize(None, None, None, 3.2)
    assert r.status == "tracking" and slam._dr is None


def test_merge_carries_distinctness_without_constraint_errors(tmp_path):
    """Regression (navigation run crash): both identities were already known to be
    distinct from a third one; re-pointing the pair hit a UNIQUE constraint."""
    memory = ExperienceMemory(tmp_path / "m.sqlite")
    for eid, t in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        memory.upsert_entity(eid, appearance={"hue": 130, "fill": 0.9}, now=t)
    memory.note_distinct("a", "c", 4.0)
    memory.note_distinct("b", "c", 4.0)
    memory.merge_entities("a", "b", now=5.0, reason="test")
    assert memory.are_distinct("a", "c") and memory.counts()["entities"] == 2
    with pytest.raises(ValueError):
        memory.merge_entities("a", "c", now=6.0, reason="test")


# ----------------------------------------------------------------- kd-tree under flush-to-zero
_FTZ_KDTREE = r"""
import ctypes, sys
import numpy as np
from amr_rl.perception.vslam import kd_tree

class FEnv(ctypes.Structure):  # macOS arm64 fenv_t
    _fields_ = [("fpsr", ctypes.c_ulonglong), ("fpcr", ctypes.c_ulonglong)]

libc = ctypes.CDLL(None)
env = FEnv()
libc.fegetenv(ctypes.byref(env))
env.fpcr |= 1 << 24  # FPCR.FZ: what Genesis' runtime leaves set on the main thread
libc.fesetenv(ctypes.byref(env))
assert not np.finfo(float).tiny / 4 > 0, "flush-to-zero not active"
points = np.load(sys.argv[1])["points"]
tree = kd_tree(points)
d, _ = tree.query(points[:50], k=1)
assert np.all(d == 0)
print("ok")
"""


@pytest.mark.skipif(not (sys.platform == "darwin" and platform.machine() == "arm64"),
                    reason="fenv_t layout and the observed failure are macOS arm64")
def test_landmark_kd_tree_builds_under_flush_to_zero():
    # Real landmark positions on which SciPy's balanced build recursed until the
    # stack overflowed once Genesis had enabled flush-to-zero (see vslam.kd_tree).
    data = Path(__file__).parent / "data" / "kdtree_ftz_landmarks.npz"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    run = subprocess.run([sys.executable, "-c", _FTZ_KDTREE, str(data)], env=env, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0 and run.stdout.strip() == "ok", run.stderr[-2000:]
