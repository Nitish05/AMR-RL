"""Floor mask from the monocular depth model (perception/near_depth.py), with a stub
depth backend that returns synthetic disparity: no model weights, no simulator."""

import numpy as np
import pytest

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.near_depth import (
    DepthConfig,
    FloorMaskConfig,
    LazyBackend,
    MonoDepthObstacles,
    load_backend,
    resolve_device,
)
from amr_rl.robot.spec import RobotSpec


@pytest.fixture(scope="module")
def cam():
    return CameraModel.from_spec(RobotSpec.load())


def _depths(det, cam, face=None):
    """Camera-frame depth per pixel (flattened): floor, optionally a vertical box face at
    base x = face[0] spanning |y| <= face[1] and height <= face[2]; far beyond the horizon.
    Returns (z, on_face)."""
    z = np.where(det.floor_eval, 1.0 / det.inv_floor, 20.0)
    on_face = np.zeros(len(z), bool)
    if face is not None:
        x0, half_w, top = face
        d = det.rays @ det.R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (x0 - det.t[0]) / d[:, 0]
        h = det.t[2] + cam.base_z + s * d[:, 2]
        y = det.t[1] + s * d[:, 1]
        on_face = (s > 0) & (h >= 0) & (h <= top) & (np.abs(y) <= half_w) & (s < z)
        z = np.where(on_face, s, z)
    return z, on_face


def _disparity(det, cam, z, a=2.0, b=0.3, noise=0.0, seed=0):
    d = a / z + b
    if noise:
        d = d * (1 + noise * np.random.default_rng(seed).standard_normal(d.shape))
    return d.reshape(cam.height, cam.width).astype(np.float32)


class StubBackend:
    """Returns the queued disparity for frame k (frames are images filled with value k)."""

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def __call__(self, rgb):
        self.calls += 1
        return self.frames[int(rgb[0, 0, 0])]


def _frame(cam, k):
    return np.full((cam.height, cam.width, 3), k, np.uint8)


def test_pure_floor_is_not_masked(cam):
    det = MonoDepthObstacles(cam)
    z, _ = _depths(det, cam)
    det.backend = StubBackend([_disparity(det, cam, z, noise=0.01)])
    not_floor, diag = det.floor_mask(_frame(cam, 0))
    assert diag["valid"] and diag["fresh"] and not diag["stale"]
    assert diag["inliers"] >= det.floor_cfg.min_inliers
    assert diag["a"] == pytest.approx(2.0, rel=0.03)
    judged = det.floor_eval.reshape(not_floor.shape)
    assert judged.sum() > 20000 and not not_floor[judged].any()
    assert not_floor[det.never_floor.reshape(not_floor.shape)].all()  # at/above the horizon
    assert diag["masked_fraction"] == 0.0


def test_near_box_face_is_masked_floor_is_not(cam):
    det = MonoDepthObstacles(cam)
    z, face = _depths(det, cam, face=(0.3, 0.1, 0.25))
    det.backend = StubBackend([_disparity(det, cam, z)])
    rgb = _frame(cam, 0)
    not_floor, diag = det.floor_mask(rgb)
    assert diag["fresh"]
    flat = not_floor.ravel()
    clearly_near = face & det.floor_eval & (z < 0.8 / det.inv_floor)
    assert clearly_near.sum() > 1000
    assert flat[clearly_near].mean() > 0.99
    floor = det.floor_eval & ~face
    assert flat[floor].mean() < 0.005
    # Point lookup (VSLAM keyframe interface), one point on the face and one on open floor.
    v, u = np.divmod(np.flatnonzero(clearly_near)[len(np.flatnonzero(clearly_near)) // 2], cam.width)
    vf, uf = np.divmod(np.flatnonzero(floor & (det.uv[:, 0] < 40))[-1], cam.width)
    pts = np.array([[u + 0.3, v - 0.2], [uf, vf]], float)
    assert det.mask_points(rgb, pts).tolist() == [True, False]
    assert det.backend.calls == 1  # mask and point lookup share one inference


def test_affine_changes_of_the_disparity_are_absorbed(cam):
    det = MonoDepthObstacles(cam)
    z, face = _depths(det, cam, face=(0.35, 0.12, 0.3))
    params = [(2.0, 0.3), (2.04, 0.31), (7.5, -1.2), (0.8, 1.0)]
    det.backend = StubBackend([_disparity(det, cam, z, a, b) for a, b in params])
    clearly_near = face & det.floor_eval & (z < 0.8 / det.inv_floor)
    floor = det.floor_eval & ~face
    for k, (a, _b) in enumerate(params):
        not_floor, diag = det.floor_mask(_frame(cam, k))
        flat = not_floor.ravel()
        assert diag["fresh"], k
        assert flat[clearly_near].mean() > 0.99 and flat[floor].mean() < 0.005, k
        if k == 1:  # small change: smoothed, not restarted
            assert not diag["ema_reset"] and 2.0 < diag["a"] < 2.04 and diag["a_frame"] > diag["a"]
        if k >= 2:  # a different model scale restarts the running fit
            assert diag["ema_reset"] and diag["a"] == pytest.approx(a, rel=0.02)


def test_too_little_floor_holds_the_last_fit_and_flags_it(cam):
    det = MonoDepthObstacles(cam)
    z0, _ = _depths(det, cam, face=(0.4, 0.1, 0.25))
    wall, on_wall = _depths(det, cam, face=(0.2, 5.0, 1.0))  # wall fills the lower image
    frames = [_disparity(det, cam, z0), _disparity(det, cam, wall), _disparity(det, cam, wall)]
    det.backend = StubBackend(frames)
    _, d0 = det.floor_mask(_frame(cam, 0))
    assert d0["fresh"]
    for k in (1, 2):
        not_floor, d = det.floor_mask(_frame(cam, k))
        assert d["valid"] and d["stale"] and not d["fresh"] and d["stale_frames"] == k
        assert (d["a"], d["b"]) == (d0["a"], d0["b"])
        near = on_wall & det.floor_eval & (wall < 0.8 / det.inv_floor)
        assert near.sum() > 10000 and not_floor.ravel()[near].mean() > 0.99
    # Without any earlier fit nothing below the horizon is masked, and it says so.
    fresh = MonoDepthObstacles(cam, backend=StubBackend(frames))
    not_floor, d = fresh.floor_mask(_frame(cam, 1))
    assert not d["valid"] and d["stale"] and d["a"] is None
    assert not not_floor.ravel()[fresh.floor_eval].any()


def test_infer_cache_is_per_frame_and_sees_in_place_changes(cam):
    det = MonoDepthObstacles(cam)
    z, _ = _depths(det, cam, face=(0.5, 0.15, 0.3))
    det.backend = StubBackend([_disparity(det, cam, z), _disparity(det, cam, z, 3.0, 0.1)])
    rgb = _frame(cam, 0)
    out = det.detect(rgb)
    det.floor_mask(rgb)
    det.mask_points(rgb, np.array([[160.0, 200.0]]))
    assert det.backend.calls == 1 and det.inferences == 1
    ref = det.detect(None, rel=det.backend.frames[0])
    np.testing.assert_array_equal(out["points"], ref["points"])
    rgb[:] = 1  # camera buffer reused in place: a new frame
    det.floor_mask(rgb)
    assert det.backend.calls == 2
    # A model output at another resolution is resampled to the camera image.
    small = MonoDepthObstacles(cam, backend=lambda img: np.ones((120, 160), np.float32))
    assert small.infer(_frame(cam, 0)).shape == (cam.height, cam.width)


def test_defaults_are_unchanged(cam):
    det = MonoDepthObstacles(cam)
    assert det.cfg == DepthConfig() and det.backend is None and not det.available
    assert det.detect(None) is None
    assert det.floor_cfg == FloorMaskConfig()
    assert LazyBackend().requested_device == "cpu" and LazyBackend().device == "cpu"
    import inspect

    assert inspect.signature(load_backend).parameters["device"].default == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")
    assert resolve_device("auto") in ("cpu", "mps", "cuda")
