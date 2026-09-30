"""Pure perception/identity/outcome contracts (no simulator)."""

import math
import threading
import time

import numpy as np
import pytest

from amr_rl.learning.identity import EntityTracker
from amr_rl.learning.memory import ExperienceMemory
from amr_rl.learning.outcomes import classify
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
