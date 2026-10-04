"""Synthetic in-place-turn scene for the turn-drift tests (no simulator, no images).

A square room (floor points, wall points up to 0.9 m) and optionally a box whose
front face stands ``box_dist`` m in front of the turning axis (points 1-35 cm high,
i.e. below the camera, so the floor-plane lifting misplaces them, as in the
round-6 drift analysis). Keypoints are the projections of the visible points from
the TRUE pose (occlusion by the box included) with unique descriptors; the truth
turns at ``slip`` x the commanded rate. ``PlanarVSLAM.features`` is replaced, so
the whole tracking, keyframe and landmark pipeline runs unchanged.
"""

import math

import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.vslam import PlanarVSLAM, VSLAMConfig
from amr_rl.robot.spec import RobotSpec


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def scene(rng, box_dist=0.35, room=4.0):
    n = 6000
    r = np.sqrt(rng.uniform(0.04, 1, n)) * room / 2
    a = rng.uniform(-math.pi, math.pi, n)
    floor = np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros(n)])
    walls = []
    for k in range(4):
        m = 500
        s, z = rng.uniform(-room / 2, room / 2, m), rng.uniform(0.02, 0.9, m)
        side = [(s, np.full(m, room / 2)), (s, np.full(m, -room / 2)),
                (np.full(m, room / 2), s), (np.full(m, -room / 2), s)][k]
        walls.append(np.column_stack([side[0], side[1], z]))
    pts, box = [floor] + walls, None
    if box_dist is not None:
        m = 400
        box = (box_dist, box_dist + 0.4, -0.2, 0.2, 0.35)
        pts.append(np.column_stack([np.full(m, box_dist), rng.uniform(-0.2, 0.2, m), rng.uniform(0.01, 0.35, m)]))
    P = np.vstack(pts)
    return P, rng.integers(0, 256, (len(P), 32), dtype=np.uint8), box


def visible(model, pose, P, box):
    uv, z = model.project_world(P, pose)
    ok = (z > 0.05) & (uv[:, 0] > 3) & (uv[:, 0] < model.width - 3) & (uv[:, 1] > 3) & (uv[:, 1] < model.height - 3)
    if box is not None:
        cam = model.T_world_cam(pose)[:3, 3]
        x0, x1, y0, y1, h = box
        d = P[:, :2] - cam[:2]
        tmin, tmax = np.zeros(len(P)), np.full(len(P), 0.999)
        for axis, lo, hi in ((0, x0 + 1e-3, x1), (1, y0, y1)):
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - cam[axis]) / d[:, axis], (hi - cam[axis]) / d[:, axis]
            ta, tb = np.minimum(t1, t2), np.maximum(t1, t2)
            par = d[:, axis] == 0
            inside = (cam[axis] >= lo) & (cam[axis] <= hi)
            ta = np.where(par, np.where(inside, -np.inf, np.inf), ta)
            tb = np.where(par, np.where(inside, np.inf, -np.inf), tb)
            tmin, tmax = np.maximum(tmin, ta), np.minimum(tmax, tb)
        zc = cam[2] + tmin * (P[:, 2] - cam[2])
        ok &= ~((tmin <= tmax) & (zc < h))
    return np.flatnonzero(ok), uv


def run_turn(cfg=None, *, box_dist=0.35, slip=0.83, w_cmd=0.45, turn_deg=360.0, seed=0, start_heading=0.0,
             base_drift=(0.0, 0.0), slam=None, inject=None):
    """Turn in place; returns (slam, rows). rows: (t, status, heading error deg relative to
    the start, position error m, result). ``base_drift``: true base velocity (m/s).
    ``inject(k, slam)``: called before frame k (tests inject estimation errors)."""
    rng = np.random.default_rng(seed)
    model = CameraModel.from_spec(RobotSpec.load())
    P, desc, box = scene(rng, box_dist)
    slam = slam or PlanarVSLAM(model, cfg or VSLAMConfig())
    truth = np.array([0.0, 0.0, start_heading])
    state = {"pose": truth}

    def features(_rgb):
        idx, uv = visible(model, state["pose"], P, box)
        return np.zeros((model.height, model.width), np.uint8), uv[idx] + rng.normal(0, 0.3, (len(idx), 2)), desc[idx]

    slam.features = features
    dt, t, prev, rows = 0.1, 0.0, (0.0, 0.0), []
    n = int(round(math.radians(turn_deg) / (slip * w_cmd * dt)))
    start = None
    for k, cmd in enumerate([(0.0, 0.0)] * 5 + [(0.0, w_cmd)] * n + [(0.0, 0.0)] * 3):
        if inject is not None:
            inject(k, slam)
        truth = truth + np.array([base_drift[0] * dt, base_drift[1] * dt, slip * prev[1] * dt])
        state["pose"] = truth
        r = slam.track(None, t, commanded=prev)
        if r.pose is not None and start is None:
            start = (r.pose.copy(), truth.copy())
        if r.pose is None or start is None:
            rows.append((t, r.status, None, None, r))
        else:
            e = wrap((r.pose[2] - start[0][2]) - (truth[2] - start[1][2]))
            c, s = math.cos(start[0][2] - start[1][2]), math.sin(start[0][2] - start[1][2])
            dtr = truth[:2] - start[1][:2]
            exp_xy = start[0][:2] + np.array([c * dtr[0] - s * dtr[1], s * dtr[0] + c * dtr[1]])
            rows.append((t, r.status, math.degrees(e), float(np.hypot(*(r.pose[:2] - exp_xy))), r))
        prev, t = cmd, t + 0.1
    return slam, rows
