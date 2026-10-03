"""Global relocalisation against a saved map (synthetic maps, no simulator).

The method (planar two-point RANSAC on floor matches, guided verification,
distinctiveness, motion-consistent confirmation over a change of view, probation)
is described in docs/VSLAM.md; the probe numbers are in docs/results.
"""

import math

import numpy as np
import pytest

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.vslam import RELOCALIZING, TRACKING, PlanarVSLAM, VSLAMConfig
from amr_rl.robot.spec import RobotSpec


@pytest.fixture(scope="module")
def model():
    return CameraModel.from_spec(RobotSpec.load())


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def synthetic_map(model, rng, n_floor=4000, n_wall=400, size=4.0):
    floor = np.column_stack([rng.uniform(-size / 2, size / 2, (n_floor, 2)), np.zeros(n_floor)])
    wall = np.column_stack([rng.uniform(-size / 2, size / 2, n_wall), np.full(n_wall, size / 2),
                            rng.uniform(0.05, 0.6, n_wall)])
    pos = np.vstack([floor, wall])
    desc = rng.integers(0, 256, (len(pos), 32), dtype=np.uint8)
    kind = np.r_[np.zeros(n_floor, np.int8), np.ones(n_wall, np.int8)]
    return pos, desc, kind


def map_slam(model, pos, desc, kind, **cfg):
    slam = PlanarVSLAM(model, VSLAMConfig(**cfg))
    slam.lm.add(pos, desc, 0, confirmed=True)
    slam.lm.kind[:] = kind
    slam.status = RELOCALIZING
    return slam


def frame_from(model, pose, pos, desc, rng, *, inlier_ratio, n_correct=150, max_range=2.0):
    """Keypoints of landmarks visible from ``pose`` (correct matches) plus keypoints at
    random pixels carrying descriptors of landmarks that are NOT there (wrong matches)."""
    uv, z = model.project_world(pos, pose)
    rng_xy = np.hypot(*(pos[:, :2] - pose[:2]).T)
    vis = np.flatnonzero((z > 0.1) & (uv[:, 0] > 2) & (uv[:, 0] < model.width - 2) & (uv[:, 1] > 2)
                         & (uv[:, 1] < model.height - 2) & (rng_xy < max_range))
    vis = rng.permutation(vis)[:n_correct]
    others = np.setdiff1d(np.arange(len(pos)), vis)
    n_wrong = int(round(len(vis) * (1 - inlier_ratio) / inlier_ratio))
    wrong = rng.choice(others, n_wrong, replace=False)
    pts = np.vstack([uv[vis] + rng.normal(0, 0.5, (len(vis), 2)),
                     np.column_stack([rng.uniform(0, model.width, n_wrong), rng.uniform(0, model.height, n_wrong)])])
    return pts, np.vstack([desc[vis], desc[wrong]])


@pytest.mark.parametrize("ratio", [0.2, 0.1])
def test_planar_relocalisation_recovers_pose_at_low_inlier_ratios(model, ratio):
    """The legacy PnP-RANSAC on (mostly coplanar) floor points failed below ~30 %
    correct matches; two-point planar hypotheses do not need large clean subsets."""
    rng = np.random.default_rng(1)
    pos, desc, kind = synthetic_map(model, rng)
    ok = 0
    for trial in range(30):
        truth = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.3), rng.uniform(-math.pi, math.pi)])
        pts, d = frame_from(model, truth, pos, desc, rng, inlier_ratio=ratio)
        slam = map_slam(model, pos, desc, kind)
        slam.frames = trial
        out = slam.global_localize(pts, d)
        if out is not None and np.hypot(*(out[0][:2] - truth[:2])) < 0.02 and abs(wrap(out[0][2] - truth[2])) < math.radians(1):
            ok += 1
    assert ok >= 29


def test_a_copied_floor_patch_is_ambiguous_not_a_pose(model):
    """Repetitive texture: the same descriptors at two places 0.8 m apart must not
    yield a pose (the round-3 seed-4 failure was an alias of this kind)."""
    rng = np.random.default_rng(2)
    pos, desc, kind = synthetic_map(model, rng, n_wall=0)
    truth = np.array([0.0, -0.5, math.pi / 2])
    patch = np.flatnonzero(np.hypot(pos[:, 0] - 0.0, pos[:, 1] - 0.4) < 0.9)
    alias = pos[patch] + np.array([0.8, 0.0, 0.0])
    pos2 = np.vstack([pos, alias])
    desc2 = np.vstack([desc, desc[patch]])
    kind2 = np.r_[kind, kind[patch]]
    pts, d = frame_from(model, truth, pos, desc, rng, inlier_ratio=1.0)
    slam = map_slam(model, pos2, desc2, kind2)
    out = slam.global_localize(pts, d)
    # Never the copy: either no pose, or the true one (landmarks outside the patch
    # are unique and can still identify the place).
    assert out is None or np.hypot(*(out[0][:2] - truth[:2])) < 0.05


def test_planar_relocalisation_is_deterministic(model):
    rng = np.random.default_rng(3)
    pos, desc, kind = synthetic_map(model, rng)
    pts, d = frame_from(model, np.array([0.2, -0.3, 1.0]), pos, desc, rng, inlier_ratio=0.2)
    a = map_slam(model, pos, desc, kind).global_localize(pts, d)
    b = map_slam(model, pos, desc, kind).global_localize(pts, d)
    assert a is not None and b is not None and np.allclose(a[0], b[0])


def chained(slam, poses, commanded, dt, t0=0.0):
    calls = iter([(p, 120, np.eye(3)) for p in poses])
    slam.global_localize = lambda pts, desc, rgb=None: next(calls)
    out = []
    for k in range(len(poses)):
        out.append(slam._relocalize(None, None, None, t0 + dt * (k + 1), commanded=commanded, dt=dt))
    return out


def test_identical_views_never_confirm_a_relocalisation(model):
    """Two (or ten) near-identical frames confirm an aliased match as readily as a
    true one: without a change of view, nothing is accepted."""
    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(4)))
    pose = np.array([0.1, 0.2, 0.3])
    results = chained(slam, [pose] * 10, (0.0, 0.0), 0.1)
    assert all(r.status != TRACKING for r in results)


def test_confirmation_follows_the_commanded_turn(model):
    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(5)))
    step = math.radians(4)
    turning = [np.array([0.1, 0.2, 0.3 + step * k]) for k in range(8)]
    results = chained(slam, turning, (0.0, 0.4), step / 0.4)
    first = next(i for i, r in enumerate(results) if r.status == TRACKING)
    assert first >= 5  # >= 20 deg of commanded rotation across the chain
    assert slam._probation == slam.cfg.reloc_probation_frames
    # a candidate that does not follow the commanded turn restarts the chain
    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(5)))
    jumpy = [np.array([0.1, 0.2, 0.3 + step * k]) for k in range(4)] + [np.array([0.9, -0.4, 2.0])] + \
            [np.array([0.9, -0.4, 2.0 + step * k]) for k in range(1, 4)]
    results = chained(slam, jumpy, (0.0, 0.4), step / 0.4)
    assert all(r.status != TRACKING for r in results)


def test_failure_during_probation_returns_to_relocalising_without_map_growth(model):
    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(6)))
    step = math.radians(4)
    chained(slam, [np.array([0.1, 0.2, 0.3 + step * k]) for k in range(6)], (0.0, 0.4), step / 0.4)
    assert slam.status == TRACKING and slam._probation > 0
    n = len(slam.lm.pos)
    blank = np.zeros((model.height, model.width, 3), np.uint8)  # nothing to track
    r = slam.track(blank, 10.0, commanded=(0.0, 0.0))
    assert r.status == RELOCALIZING and r.reason == "relocalization_rejected_in_probation"
    assert len(slam.lm.pos) == n and slam._probation == 0


def test_legacy_method_remains_selectable(model):
    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(7)), reloc_method="pnp")
    assert slam.global_localize(np.zeros((0, 2)), np.zeros((0, 32), np.uint8)) is None


def test_map_schema_v2_round_trips_anchors_and_place_descriptors(model, tmp_path):
    from amr_rl.perception.vslam import Keyframe

    slam = map_slam(model, *synthetic_map(model, np.random.default_rng(8)))
    slam.lm.anchor[:] = np.arange(len(slam.lm.anchor)) % 3
    for i in range(3):
        kf = Keyframe(i, float(i), np.array([0.1 * i, 0.0, 0.0]), np.zeros((0, 2)), np.zeros((0, 32), np.uint8),
                      np.zeros(0, int))
        kf.gdesc = np.random.default_rng(i).normal(size=16).astype(np.float32)
        kf.gdesc /= np.linalg.norm(kf.gdesc)
        slam.keyframes.append(kf)
    slam.place = type("P", (), {"name": "test"})()
    meta = slam.save(tmp_path)
    assert meta["schema"] == "amr_rl.vslam-map.v2" and meta["place_descriptor"] == "test"
    back = PlanarVSLAM.load(tmp_path, model)
    assert np.array_equal(back.lm.anchor, slam.lm.anchor)
    assert all(np.allclose(a.gdesc, b.gdesc, atol=1e-3) for a, b in zip(slam.keyframes, back.keyframes))
    assert back.n_loaded_keyframes == 3
