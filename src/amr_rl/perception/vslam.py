"""RGB-only planar visual SLAM for a ground robot with one forward camera.

Pipeline (see docs/VSLAM.md for assumptions and validation):

1. ORB features on CLAHE-normalised grayscale.
2. Tracking: constant-velocity prediction, guided descriptor matching of map
   landmarks projected into the image, robust (Huber) Gauss-Newton on the planar
   pose (x, y, theta) minimising reprojection error.
3. Mapping: keyframes on motion; new landmarks by two-view triangulation between
   keyframes with known poses, plus floor landmarks by inverse perspective
   mapping (IPM) of features below the horizon. Floor landmarks carry metric
   scale from the calibrated camera height; this is the only scale source.
4. Relocalization: global descriptor matching + PnP-RANSAC, accepted only when
   the solution is planar-consistent and repeated on consecutive frames.

Monocular scale ambiguity is resolved by the flat-floor + known-camera-height
assumption, not by any simulator quantity. What this module does NOT do: loop
closure/pose-graph optimisation, bundle adjustment, dynamic-object handling or
obstacle mapping (see ``amr_rl.mapping``). A sparse landmark map is not an
obstacle map.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .camera_model import CameraModel, wrap

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def kd_tree(points) -> cKDTree:
    """Nearest-neighbour index over ``points``.

    ``balanced_tree=False`` (sliding-midpoint splits) is required, not a tuning
    choice: Genesis' runtime enables flush-to-zero on the main thread, and under it
    SciPy's median-split build recursed without end on real landmark maps (macOS
    arm64, SciPy 1.18.1: segfault by stack overflow). Sliding-midpoint splits always
    separate at least one point, so the build terminates. Queries are exact either
    way; only the tree shape differs. Regression: ``tests/test_perception_units.py``.
    """
    return cKDTree(points, balanced_tree=False)
TRACKING, LOST, RELOCALIZING, INITIALIZING = "tracking", "lost", "relocalizing", "initializing"
PREDICTED = "predicted"  # bounded dead reckoning after brief visual loss (<= 1.5 s, <= 8 cm sigma)


@dataclass
class VSLAMConfig:
    n_features: int = 900
    fast_threshold: int = 10
    match_radius: float = 22.0
    wide_radius: float = 50.0
    max_hamming: int = 56
    ratio: float = 0.8
    min_inliers: int = 22
    huber_px: float = 2.0
    inlier_px: float = 4.0
    kf_translation: float = 0.10
    kf_rotation: float = math.radians(10)
    kf_min_inliers: int = 90
    ipm_max_range: float = 2.2
    ipm_margin_px: float = 10.0
    min_parallax_deg: float = 1.5
    max_landmarks: int = 60000
    lost_after_failures: int = 3
    max_prediction_seconds: float = 1.5
    max_prediction_sigma: float = 0.08
    max_prediction_frames: int = 10
    reloc_min_inliers: int = 30
    reloc_confirmations: int = 2
    motion_prior: bool = True
    consistency_window: float = 3.0
    # Fast "frozen estimate" check: commanded travel >= freeze_min_cmd within
    # freeze_window while vision reports < freeze_ratio of it -> wrong lock.
    freeze_window: float = 1.2
    static_hypothesis_when_moving: bool = True  # False was evaluated and rejected (docs/VSLAM.md)
    command_consistency_score: bool = False  # evaluated: no measurable effect (docs/VSLAM.md)
    reloc_local_first: bool = False  # evaluated: no effect (re-acquisition never succeeded first)
    # Loss handling added in round 2 (switchable so the pre-change behaviour can be benchmarked).
    freeze_check: bool = True
    purge_on_loss: bool = True
    dr_gate: bool = True
    freeze_min_cmd: float = 0.12
    freeze_ratio: float = 0.25
    # After a motion-inconsistency loss, relocalisation must agree with dead reckoning
    # from the last trustworthy pose (gate grows with commanded travel) for this long.
    reloc_dr_gate_seconds: float = 20.0
    local_ba_window: int = 6
    local_ba_fixed: int = 2
    sigma_scale: float = 2.0
    sigma_floor: float = 0.005


@dataclass
class Keyframe:
    id: int
    timestamp: float
    pose: np.ndarray
    pts: np.ndarray
    desc: np.ndarray
    landmark: np.ndarray  # landmark index per feature or -1
    gray: np.ndarray | None = None


@dataclass
class TrackResult:
    status: str
    pose: np.ndarray | None
    position_sigma: float | None
    heading_sigma: float | None
    inliers: int
    matched: int
    keyframe: bool
    timestamp: float
    reason: str = ""
    inlier_uv: np.ndarray | None = None
    info: dict = field(default_factory=dict)


class Landmarks:
    def __init__(self):
        self.pos = np.zeros((0, 3))
        self.desc = np.zeros((0, 32), np.uint8)
        self.kind = np.zeros(0, np.int8)  # 0 floor-IPM, 1 triangulated
        self.visible = np.zeros(0, np.int32)
        self.found = np.zeros(0, np.int32)
        self.alive = np.zeros(0, bool)
        self.confirmed = np.zeros(0, bool)
        self.origin = np.zeros((0, 3))  # camera centre at creation
        self.marked = np.zeros(0, bool)  # already contributed obstacle evidence
        self.created = np.zeros(0)  # timestamp of creation (-inf: loaded from a saved map)

    def __len__(self):
        return int(self.alive.sum())

    def add(self, pos, desc, kind, origin=None, confirmed=False, created=-np.inf):
        n = len(pos)
        if n == 0:
            return np.zeros(0, int)
        start = len(self.pos)
        self.pos = np.vstack([self.pos, pos])
        self.desc = np.vstack([self.desc, desc])
        self.kind = np.concatenate([self.kind, np.full(n, kind, np.int8)])
        self.visible = np.concatenate([self.visible, np.zeros(n, np.int32)])
        self.found = np.concatenate([self.found, np.zeros(n, np.int32)])
        self.alive = np.concatenate([self.alive, np.ones(n, bool)])
        self.confirmed = np.concatenate([self.confirmed, np.full(n, bool(confirmed))])
        self.marked = np.concatenate([self.marked, np.zeros(n, bool)])
        o = np.zeros((n, 3)) if origin is None else np.broadcast_to(np.asarray(origin, float), (n, 3))
        self.origin = np.vstack([self.origin, o])
        self.created = np.concatenate([self.created, np.full(n, float(created))])
        return np.arange(start, start + n)

    def compact(self):
        keep = self.alive
        remap = -np.ones(len(keep), int)
        remap[keep] = np.arange(int(keep.sum()))
        for name in ("pos", "desc", "kind", "visible", "found", "alive", "confirmed", "origin", "marked", "created"):
            setattr(self, name, getattr(self, name)[keep])
        return remap


class PlanarVSLAM:
    def __init__(self, model: CameraModel, config: VSLAMConfig | None = None, *, map_version=None):
        self.model = model
        self.cfg = config or VSLAMConfig()
        self.orb = cv2.ORB_create(nfeatures=self.cfg.n_features, scaleFactor=1.2, nlevels=6,
                                  edgeThreshold=15, patchSize=15, fastThreshold=self.cfg.fast_threshold)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lm = Landmarks()
        self.keyframes: list[Keyframe] = []
        self.status = INITIALIZING
        self.pose: np.ndarray | None = None
        self.velocity = np.zeros(3)
        self.last_time: float | None = None
        self.failures = 0
        self.position_sigma: float | None = None
        self.heading_sigma: float | None = None
        self.map_version = map_version or f"map-{uuid.uuid4().hex[:10]}"
        self._reloc_candidates: list[np.ndarray] = []
        self.horizon = model.horizon_row()
        self.frames = 0
        self.last_hypothesis = None
        self.predicted_time = 0.0
        self._kp_tree = self._kp_tree_pts = None
        self._frame_candidates = None
        self._T_cb = model.T_cam_base
        self._motion_prior = None
        self._motion_log = []
        self._now = 0.0
        self._dr = None  # dead-reckoning gate after a loss of tracking
        self.inconsistency = None
        self.frozen_events = []
        self._last_err = None
        self._last_quality = (0, 0.0)
        self.distance_travelled = 0.0

    # ------------------------------------------------------------ features
    def features(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = self.clahe.apply(gray)
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or not kps:
            return gray, np.zeros((0, 2)), np.zeros((0, 32), np.uint8)
        pts = np.array([k.pt for k in kps], float)
        return gray, pts, desc

    # ------------------------------------------------------------ geometry
    def _project(self, P, pose):
        return self.model.project_world(P, pose)

    def _residuals(self, pose, P, uv):
        proj, z = self._project(P, pose)
        r = (proj - uv).ravel()
        r[np.repeat(z <= 0.02, 2)] = 1e3
        return r

    def _residuals_jacobian(self, pose, P, uv):
        """Reprojection residuals (2N) and analytic Jacobian (2N x 3) for planar pose."""
        x, y, th = pose
        c, s_ = math.cos(th), math.sin(th)
        Rt = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
        dRt = np.array([[-s_, c, 0.0], [-c, -s_, 0.0], [0.0, 0.0, 0.0]])
        T_cb = self._T_cb
        R_cb, t_cb = T_cb[:3, :3], T_cb[:3, 3]
        d = P - np.array([x, y, self.model.base_z])
        pb = d @ Rt.T
        pc = pb @ R_cb.T + t_cb
        X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
        Zs = np.where(Z > 0.02, Z, 0.02)
        fx, fy, cx, cy = self.model.K[0, 0], self.model.K[1, 1], self.model.K[0, 2], self.model.K[1, 2]
        r = np.empty((len(P), 2))
        r[:, 0] = fx * X / Zs + cx - uv[:, 0]
        r[:, 1] = fy * Y / Zs + cy - uv[:, 1]
        bad = Z <= 0.02
        r[bad] = 1e3
        # derivatives of p_c wrt (x, y, theta)
        dpc_dx = -(R_cb @ Rt[:, 0])
        dpc_dy = -(R_cb @ Rt[:, 1])
        dpc_dth = (d @ dRt.T) @ R_cb.T
        J = np.zeros((len(P), 2, 3))
        for k, dpc in enumerate((np.broadcast_to(dpc_dx, pc.shape), np.broadcast_to(dpc_dy, pc.shape), dpc_dth)):
            J[:, 0, k] = fx * (dpc[:, 0] / Zs - X * dpc[:, 2] / Zs ** 2)
            J[:, 1, k] = fy * (dpc[:, 1] / Zs - Y * dpc[:, 2] / Zs ** 2)
        J[bad] = 0.0
        return r.ravel(), J.reshape(-1, 3)

    def optimize(self, pose0, P, uv, iters=10, weights=None, prior=None):
        """Robust planar pose fit. ``prior`` = (pose, sigmas): the robot's own commanded
        motion as a weak Gaussian prior (whitened, Huber-robust); vision dominates
        whenever it is informative."""
        x = np.asarray(pose0, float).copy()
        H = np.eye(3)
        for _ in range(iters):
            r, J = self._residuals_jacobian(x, P, uv)
            a = np.abs(r)
            w = np.where(a <= self.cfg.huber_px, 1.0, self.cfg.huber_px / np.maximum(a, 1e-9))
            if weights is not None:
                w = w * np.repeat(weights, 2)
            H = J.T @ (J * w[:, None])
            g = J.T @ (w * r)
            if prior is not None:
                p_pose, sig = prior
                d = np.array([x[0] - p_pose[0], x[1] - p_pose[1], wrap(x[2] - p_pose[2])]) / sig
                wp = np.where(np.abs(d) <= 3.0, 1.0, 3.0 / np.maximum(np.abs(d), 1e-9))
                H = H + np.diag(wp / sig ** 2)
                g = g + wp * d / sig
            try:
                dx = -np.linalg.solve(H + 1e-6 * np.eye(3), g)
            except np.linalg.LinAlgError:
                break
            x += dx
            x[2] = wrap(x[2])
            if np.abs(dx[:2]).max() < 1e-5 and abs(dx[2]) < 1e-6:
                break
        r, _ = self._residuals_jacobian(x, P, uv)
        err = np.linalg.norm(r.reshape(-1, 2), axis=1)
        return x, err, H

    # ------------------------------------------------------------ matching
    def _guided(self, pose, pts, desc, radius, count_visible=True):
        """Match map landmarks projected at ``pose`` to keypoints within ``radius`` px.
        Ratio test per landmark, one landmark per keypoint (lowest Hamming wins)."""
        empty = (np.zeros(0, int), np.zeros(0, int))
        cand = self._frame_candidates if self._frame_candidates is not None else np.flatnonzero(self.lm.alive)
        if len(cand) == 0 or len(pts) == 0:
            return empty
        proj, z = self._project(self.lm.pos[cand], pose)
        W, Hh = self.model.width, self.model.height
        inview = (z > 0.05) & (z < 8.0) & (proj[:, 0] > -5) & (proj[:, 0] < W + 5) & (proj[:, 1] > -5) & (proj[:, 1] < Hh + 5)
        cand, proj = cand[inview], proj[inview]
        if len(cand) == 0:
            return empty
        if count_visible:
            self.lm.visible[cand] += 1
        tree = self._kp_tree if self._kp_tree_pts is pts else None
        if tree is None:
            tree = kd_tree(pts)
            self._kp_tree, self._kp_tree_pts = tree, pts
        dist, idx = tree.query(proj, k=min(8, len(pts)), distance_upper_bound=radius)
        dist, idx = np.atleast_2d(dist.T).T if dist.ndim == 1 else dist, idx if idx.ndim == 2 else idx[:, None]
        valid = np.isfinite(dist)
        if not valid.any():
            return empty
        li = np.nonzero(valid)[0]
        ki = idx[valid]
        ham = _POPCOUNT[np.bitwise_xor(self.lm.desc[cand[li]], desc[ki])].sum(1)
        order = np.lexsort((ham, li))
        li, ki, ham = li[order], ki[order], ham[order]
        first = np.ones(len(li), bool)
        first[1:] = li[1:] != li[:-1]
        second = np.zeros(len(li), bool)
        second[1:] = first[:-1] & ~first[1:]
        best_idx = np.flatnonzero(first)
        second_ham = np.full(len(cand), 10_000)
        second_ham[li[second]] = ham[second]
        bl, bk, bh = li[best_idx], ki[best_idx], ham[best_idx]
        keep = (bh <= self.cfg.max_hamming) & (bh < self.cfg.ratio * second_ham[bl])
        bl, bk, bh = bl[keep], bk[keep], bh[keep]
        if len(bl) == 0:
            return empty
        order = np.lexsort((bh, bk))
        bl, bk = bl[order], bk[order]
        uniq = np.ones(len(bk), bool)
        uniq[1:] = bk[1:] != bk[:-1]
        return cand[bl[uniq]], bk[uniq]

    def _pose_from_matches(self, pose0, lm_idx, kp_idx, pts):
        P, uv = self.lm.pos[lm_idx], pts[kp_idx]
        prior = self._motion_prior
        pose, err, _ = self.optimize(pose0, P, uv, prior=prior)
        inl = err < self.cfg.inlier_px * 2
        if inl.sum() >= self.cfg.min_inliers:
            use = inl & self.lm.confirmed[lm_idx]
            if use.sum() < self.cfg.min_inliers:
                use = inl  # early map: tentative floor points are all there is
            # Established landmarks (re-observed many times) anchor the estimate more
            # strongly than fresh ones, limiting drift from newly added points.
            weights = np.clip(0.3 + self.lm.found[lm_idx[use]] / 8.0, 0.3, 1.5)
            pose, err, H = self.optimize(pose, P[use], uv[use], weights=weights, prior=prior)
            full = np.linalg.norm(self._residuals_jacobian(pose, P, uv)[0].reshape(-1, 2), axis=1)
            inl = full < self.cfg.inlier_px
            return pose, inl, H, full
        return pose, inl, None, err

    # ------------------------------------------------------------ public API
    def _agrees_with_commands(self, pose, dt):
        """Is a solution within a per-frame gate of the commanded-motion prediction?"""
        cmd_pose, sig = self._motion_prior
        d_pos = float(np.hypot(pose[0] - cmd_pose[0], pose[1] - cmd_pose[1]))
        d_th = abs(wrap(pose[2] - cmd_pose[2]))
        return d_pos <= 0.02 + 2.0 * sig[0] and d_th <= 0.03 + 2.0 * sig[2]

    def _hypotheses(self, dt, commanded):
        """Prediction hypotheses: constant velocity, static, and the robot's own
        commanded body motion (proprioceptive knowledge of its actions, not a sensor)."""
        out = []
        cv = self.pose + self.velocity * dt
        cv[2] = wrap(cv[2])
        out.append(("constant_velocity", cv))
        moving = commanded is not None and (abs(commanded[0]) >= 0.05 or abs(commanded[1]) >= 0.2)
        if not moving or self.cfg.static_hypothesis_when_moving:
            # Optionally skip "static" while driven (suspected of aliased locks on
            # repetitive texture); benchmarked worse overall, so off by default.
            out.append(("static", self.pose.copy()))
        if commanded is not None and dt > 0:
            v, w = commanded
            th = self.pose[2] + 0.5 * w * dt
            cm = self.pose + np.array([v * dt * math.cos(th), v * dt * math.sin(th), w * dt])
            cm[2] = wrap(cm[2])
            out.append(("commanded", cm))
        return out

    def track(self, rgb: np.ndarray, timestamp: float, commanded=None) -> TrackResult:
        self.frames += 1
        gray, pts, desc = self.features(rgb)
        dt = 0.0 if self.last_time is None else max(0.0, timestamp - self.last_time)
        self.last_time = timestamp
        self._now = timestamp
        if self._dr is not None and commanded is not None and dt > 0:
            v, w = commanded
            th = self._dr["pose"][2] + 0.5 * w * dt
            self._dr["pose"] = self._dr["pose"] + np.array([v * dt * math.cos(th), v * dt * math.sin(th), w * dt])
            self._dr["travel"] += abs(v) * dt + 0.1 * abs(w) * dt
            self._dr["rot"] += abs(w) * dt
        if self.status == INITIALIZING:
            return self._initialize(gray, pts, desc, timestamp)
        if self.status in (LOST, RELOCALIZING):
            return self._relocalize(gray, pts, desc, timestamp)
        # TRACKING or PREDICTED: try to (re)acquire against the local map.
        hypotheses = self._hypotheses(dt, commanded)
        self._motion_prior = None
        if commanded is not None and dt > 0 and self.cfg.motion_prior:
            cmd_pose = dict(hypotheses)["commanded"]
            v, w = commanded
            sig = np.array([0.5 * abs(v) * dt + 0.01, 0.5 * abs(v) * dt + 0.01, 0.5 * abs(w) * dt + 0.02])
            self._motion_prior = (cmd_pose, sig)
        alive = np.flatnonzero(self.lm.alive)
        dx = self.lm.pos[alive, 0] - self.pose[0]
        dy = self.lm.pos[alive, 1] - self.pose[1]
        # Only landmarks that could be in view: within 4 m and +-70 deg of the heading
        # (horizontal FOV is +-37.6 deg; the margin covers the prediction hypotheses).
        rel = np.arctan2(dy, dx) - self.pose[2]
        rel = np.arctan2(np.sin(rel), np.cos(rel))
        near = (np.hypot(dx, dy) < 4.0) & (np.abs(rel) < math.radians(70))
        self._frame_candidates = alive[near]
        result = None
        for radius in (self.cfg.match_radius, self.cfg.wide_radius):
            best = None
            for name, predicted in hypotheses:
                lm_idx, kp_idx = self._guided(predicted, pts, desc, radius)
                if len(lm_idx) < self.cfg.min_inliers:
                    continue
                pose, inl, H, err = self._pose_from_matches(predicted, lm_idx, kp_idx, pts)
                if H is None or inl.sum() < self.cfg.min_inliers or not self._plausible(pose, self.pose, dt):
                    continue
                # Refine: re-match tightly around the optimised pose, then re-optimise.
                lm2, kp2 = self._guided(pose, pts, desc, 8.0, count_visible=False)
                if len(lm2) >= self.cfg.min_inliers:
                    pose2, inl2, H2, err2 = self._pose_from_matches(pose, lm2, kp2, pts)
                    if H2 is not None and inl2.sum() >= inl.sum():
                        pose, inl, H, lm_idx, kp_idx, err = pose2, inl2, H2, lm2, kp2, err2
                score = int(inl.sum())
                if (self.cfg.command_consistency_score and self._motion_prior is not None
                        and not self._agrees_with_commands(pose, dt)):
                    score = int(0.6 * score)  # needs clearly more support than a command-consistent fit
                if best is None or score > best[0]:
                    best = (score, (pose, inl, H, lm_idx, kp_idx, err), name)
                if score >= 80 and float(np.mean(err[inl])) < 1.5:
                    break  # confident: skip the remaining prediction hypotheses
            if best is not None:
                result = best[1]
                self.last_hypothesis = best[2]
                break
        if result is None:
            self.failures += 1
            self.predicted_time += dt
            # Bounded prediction from the robot's own commanded motion (dead reckoning),
            # with growing uncertainty. Beyond the time/uncertainty limits -> LOST.
            if commanded is not None and dt > 0:
                v, w = commanded
                th = self.pose[2] + 0.5 * w * dt
                self.pose = self.pose + np.array([v * dt * math.cos(th), v * dt * math.sin(th), w * dt])
                self.pose[2] = wrap(self.pose[2])
                self.position_sigma = (self.position_sigma or 0.0) + 0.5 * abs(v) * dt + 0.002
                self.heading_sigma = (self.heading_sigma or 0.0) + 0.3 * abs(w) * dt + 0.003
            if (self.predicted_time > self.cfg.max_prediction_seconds + 1e-9
                    or (self.position_sigma or 0.0) > self.cfg.max_prediction_sigma
                    or self.failures > 2 * self.cfg.max_prediction_frames):
                # Tracking usually degrades before it fails (a wrong lock for ~1 s):
                # landmarks created in the window before the loss are suspect too.
                self._arm_dead_reckoning(timestamp, remove_landmarks=True)
                self.status = LOST
                self.position_sigma = self.heading_sigma = None
                self.velocity[:] = 0
                self._motion_log.clear()
                return TrackResult(LOST, None, None, None, 0, 0, False, timestamp, "tracking_failed")
            self.status = PREDICTED
            return TrackResult(PREDICTED, self.pose.copy(), self.position_sigma, self.heading_sigma,
                               0, 0, False, timestamp, "visual_tracking_interrupted", info={"degraded": True})
        self.status = TRACKING
        self.predicted_time = 0.0
        pose, inl, H, lm_idx, kp_idx, err = result
        self._frame_candidates = None
        if not self._motion_consistent(pose, commanded, dt, timestamp):
            self._on_frozen(timestamp)
            self.status = LOST
            self.position_sigma = self.heading_sigma = None
            self.velocity[:] = 0
            self._motion_log.clear()
            return TrackResult(LOST, None, None, None, 0, 0, False, timestamp, "visual_motion_inconsistent_with_commands")
        self.failures = 0
        if dt > 0:
            step = np.array([pose[0] - self.pose[0], pose[1] - self.pose[1], wrap(pose[2] - self.pose[2])])
            self.velocity = 0.5 * self.velocity + 0.5 * step / dt
            self.distance_travelled += float(np.hypot(*step[:2]))
        self.pose = pose
        self.lm.found[lm_idx[inl]] += 1
        centre = self.model.T_world_cam(pose)[:3, 3]
        tent = lm_idx[inl][~self.lm.confirmed[lm_idx[inl]]]
        if len(tent):
            moved = np.linalg.norm(self.lm.origin[tent] - centre, axis=1) >= 0.04
            ok = moved & (self.lm.found[tent] >= 3)
            self.lm.confirmed[tent[ok]] = True
        self._last_quality = (int(inl.sum()), float(np.mean(err[inl])))
        self._set_sigma(H, int(inl.sum()))
        matched_lm = -np.ones(len(pts), int)
        matched_lm[kp_idx[inl]] = lm_idx[inl]
        kf = self._maybe_keyframe(gray, pts, desc, matched_lm, int(inl.sum()), timestamp)
        return TrackResult(TRACKING, pose.copy(), self.position_sigma, self.heading_sigma,
                           int(inl.sum()), len(lm_idx), kf, timestamp,
                           inlier_uv=pts[kp_idx[inl]])

    def _on_frozen(self, t):
        """Vision disagreed with the commanded motion: distrust everything since the
        start of the consistency window. Landmarks created since then were placed from
        wrong poses (they would pull relocalisation back to the frozen pose), so they
        are removed; relocalisation is gated by dead reckoning from the window start."""
        self._arm_dead_reckoning(t, remove_landmarks=True, frozen=self.inconsistency == "frozen")

    def _arm_dead_reckoning(self, t, *, remove_landmarks, frozen=False):
        """On any loss of tracking: anchor dead reckoning at the oldest pose of the
        consistency window (the last one not implicated in the failure) and integrate
        the commanded motion since. Relocalisation must then agree with it in position
        and heading (gates grow with commanded travel) for reloc_dr_gate_seconds."""
        log = self._motion_log
        if not log:
            anchor_t, anchor = t, (None if self.pose is None else np.asarray(self.pose, float).copy())
            if anchor is None:
                return
        else:
            anchor_t, anchor = log[0][0], log[0][1].copy()
            if frozen:
                anchor_t = max(anchor_t, t - self.cfg.freeze_window)
                anchor = next((e[1].copy() for e in log if e[0] >= anchor_t), anchor)
        dr, travel, rot = anchor.copy(), 0.0, 0.0
        for e in log:
            if e[0] > anchor_t:
                th = dr[2] + 0.5 * e[3]
                dr = dr + np.array([e[2] * math.cos(th), e[2] * math.sin(th), e[3]])
                travel += abs(e[2]) + 0.1 * abs(e[3])
                rot += abs(e[3])
        removed = 0
        if remove_landmarks and self.cfg.purge_on_loss:
            poisoned = self.lm.alive & (self.lm.created >= anchor_t)
            self.lm.alive[poisoned] = False
            removed = int(poisoned.sum())
        if not self.cfg.dr_gate:
            self.frozen_events.append({"t": t, "window_start": anchor_t, "reason": "loss", "landmarks_removed": removed})
            return
        self._dr = {"pose": dr, "travel": travel, "rot": rot, "since": t,
                    "frozen_pose": None if self.pose is None else np.asarray(self.pose, float).copy()}
        self.frozen_events.append({"t": t, "window_start": anchor_t, "reason": "frozen" if frozen else
                                   (self.inconsistency or "tracking_failed"), "landmarks_removed": removed})

    def _motion_consistent(self, pose, commanded, dt, t):
        """Compare visual displacement with the robot's own commanded motion over a
        window. Vision reporting far less (or far more) motion than was commanded
        for seconds indicates a wrong lock (repetitive texture, degenerate view).
        Wheel slip or being blocked can also cause it; either way stopping is safe."""
        if commanded is None:
            return True
        v, w = commanded
        self._motion_log.append((t, np.asarray(pose, float).copy(), v * dt, w * dt))
        while self._motion_log and t - self._motion_log[0][0] > self.cfg.consistency_window:
            self._motion_log.pop(0)
        recent = [e for e in self._motion_log if t - e[0] <= self.cfg.freeze_window + 1e-9]
        if self.cfg.freeze_check and len(recent) >= 5 and t - recent[0][0] >= 0.8 * self.cfg.freeze_window:
            cmd_recent = sum(abs(e[2]) for e in recent[1:])
            vis_recent = float(np.hypot(*(recent[-1][1][:2] - recent[0][1][:2])))
            if cmd_recent >= self.cfg.freeze_min_cmd and vis_recent < self.cfg.freeze_ratio * cmd_recent:
                self.inconsistency = "frozen"
                return False
        if len(self._motion_log) < 10 or t - self._motion_log[0][0] < 0.8 * self.cfg.consistency_window:
            return True
        cmd_lin = sum(abs(e[2]) for e in self._motion_log[1:])
        cmd_ang = sum(e[3] for e in self._motion_log[1:])
        p0, p1 = self._motion_log[0][1], self._motion_log[-1][1]
        vis_lin = float(np.hypot(*(p1[:2] - p0[:2])))
        vis_ang = wrap(p1[2] - p0[2])
        # Lenient: wheel slip and acceleration lag make real motion smaller than
        # commanded, so only gross disagreement is flagged.
        lin_bad = (cmd_lin >= 0.3 and vis_lin < 0.3 * cmd_lin) or vis_lin > 2.0 * cmd_lin + 0.12
        ang_bad = abs(vis_ang) > 2.0 * abs(cmd_ang) + 0.35
        self.inconsistency = "linear" if lin_bad else "angular" if ang_bad else None
        return not (lin_bad or ang_bad)

    def _plausible(self, pose, previous, dt):
        # Planar robot at <= 0.4 m/s, <= 2 rad/s: reject teleport-like solutions.
        limit_xy = 0.08 + 0.6 * max(dt, 0.1)
        limit_th = 0.15 + 2.5 * max(dt, 0.1)
        return (np.hypot(pose[0] - previous[0], pose[1] - previous[1]) < limit_xy
                and abs(wrap(pose[2] - previous[2])) < limit_th)

    def _set_sigma(self, H, inliers):
        try:
            cov = np.linalg.inv(H) * (self.cfg.huber_px ** 2)
            xy = float(np.sqrt(max(np.linalg.eigvalsh(cov[:2, :2]).max(), 0)))
            th = float(np.sqrt(max(cov[2, 2], 0)))
        except np.linalg.LinAlgError:
            xy, th = 0.5, 0.5
        self.position_sigma = self.cfg.sigma_scale * xy + self.cfg.sigma_floor
        self.heading_sigma = self.cfg.sigma_scale * th + 0.002

    # ------------------------------------------------------------ mapping
    def _initialize(self, gray, pts, desc, timestamp):
        pose = np.zeros(3)
        world, ok = self._ipm(pts, pose)
        if ok.sum() < 40:
            return TrackResult(INITIALIZING, None, None, None, 0, 0, False, timestamp,
                               "insufficient_floor_texture")
        ids = self.lm.add(world[ok], desc[ok], 0, origin=self.model.T_world_cam(pose)[:3, 3], created=timestamp)
        matched = -np.ones(len(pts), int)
        matched[np.flatnonzero(ok)] = ids
        self.keyframes.append(Keyframe(0, timestamp, pose.copy(), pts, desc, matched, gray))
        self.pose = pose
        self.status = TRACKING
        self.position_sigma, self.heading_sigma = self.cfg.sigma_floor, 0.002
        return TrackResult(TRACKING, pose.copy(), self.position_sigma, self.heading_sigma,
                           int(ok.sum()), int(ok.sum()), True, timestamp, "initialized")

    def _ipm(self, pts, pose):
        if len(pts) == 0:
            return np.zeros((0, 3)), np.zeros(0, bool)
        below = pts[:, 1] > self.horizon + self.cfg.ipm_margin_px
        world, ok = self.model.ground_points_world(pts, pose, self.cfg.ipm_max_range)
        return world, ok & below

    def _maybe_keyframe(self, gray, pts, desc, matched_lm, inliers, timestamp):
        last = self.keyframes[-1]
        moved = np.hypot(*(self.pose[:2] - last.pose[:2]))
        turned = abs(wrap(self.pose[2] - last.pose[2]))
        if moved < self.cfg.kf_translation and turned < self.cfg.kf_rotation:
            # Weak tracking alone justifies a keyframe only after some motion.
            if inliers >= self.cfg.kf_min_inliers or (moved < 0.03 and turned < math.radians(3)):
                return False
        n_inl, mean_err = self._last_quality
        if n_inl < 30 or mean_err > 2.5:
            return False  # never extend the map from a weakly constrained pose
        kf = Keyframe(len(self.keyframes), timestamp, self.pose.copy(), pts, desc, matched_lm.copy(), gray)
        self._triangulate(kf, last)
        # Remaining unmatched floor-like features become metric floor landmarks.
        free = kf.landmark < 0
        world, ok = self._ipm(pts, self.pose)
        ok &= free
        if ok.any() and self.lm.alive.any():
            alive = np.flatnonzero(self.lm.alive)
            tree = kd_tree(self.lm.pos[alive])
            d, j = tree.query(world[ok], k=1, distance_upper_bound=0.02)
            dup = np.isfinite(d)
            idx = np.flatnonzero(ok)
            if dup.any():
                ham = _POPCOUNT[np.bitwise_xor(desc[idx[dup]], self.lm.desc[alive[j[dup]]])].sum(1)
                same = np.zeros(len(dup), bool)
                same[np.flatnonzero(dup)] = ham < 40
                # Only a near-identical descriptor at the same floor point is a duplicate;
                # a different appearance (scale/viewpoint) is kept as a new observation.
                ok[idx[same]] = False
        ids = self.lm.add(world[ok], desc[ok], 0, origin=self.model.T_world_cam(self.pose)[:3, 3],
                          created=self._now)
        kf.landmark[np.flatnonzero(ok)] = ids
        self.keyframes.append(kf)
        if self.cfg.local_ba_window >= 4 and len(self.keyframes) >= self.cfg.local_ba_window:
            self._local_ba()
        if len(self.keyframes) > 400:  # keep images only for recent keyframes
            self.keyframes[-400].gray = None
        self._cull()
        return True

    def _triangulate(self, kf, ref):
        a = np.flatnonzero(kf.landmark < 0)
        b = np.flatnonzero(ref.landmark < 0)
        if len(a) < 8 or len(b) < 8:
            return 0
        matches = self.matcher.knnMatch(kf.desc[a], ref.desc[b], k=2)
        pairs = [(m[0].queryIdx, m[0].trainIdx) for m in matches
                 if m and m[0].distance < 50 and (len(m) < 2 or m[0].distance < 0.8 * m[1].distance)]
        if len(pairs) < 5:
            return 0
        ia = a[[p[0] for p in pairs]]
        ib = b[[p[1] for p in pairs]]
        T1 = np.linalg.inv(self.model.T_world_cam(kf.pose))
        T2 = np.linalg.inv(self.model.T_world_cam(ref.pose))
        baseline = np.linalg.norm(np.linalg.inv(T1)[:3, 3] - np.linalg.inv(T2)[:3, 3])
        if baseline < 0.03:
            return 0
        K = self.model.K
        X = cv2.triangulatePoints(K @ T1[:3], K @ T2[:3], kf.pts[ia].T, ref.pts[ib].T)
        X = (X[:3] / X[3]).T
        uv1, z1 = self._project(X, kf.pose)
        uv2, z2 = self._project(X, ref.pose)
        e1 = np.linalg.norm(uv1 - kf.pts[ia], axis=1)
        e2 = np.linalg.norm(uv2 - ref.pts[ib], axis=1)
        c1, c2 = np.linalg.inv(T1)[:3, 3], np.linalg.inv(T2)[:3, 3]
        r1, r2 = X - c1, X - c2
        cos = (r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-9)
        parallax = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ok = ((z1 > 0.1) & (z2 > 0.1) & (e1 < 2.0) & (e2 < 2.0) & (parallax > self.cfg.min_parallax_deg)
              & (X[:, 2] > -0.05) & (X[:, 2] < 2.0) & (np.linalg.norm(r1, axis=1) < 6.0))
        ids = self.lm.add(X[ok], kf.desc[ia[ok]], 1, origin=np.linalg.inv(T1)[:3, 3], confirmed=True,
                          created=self._now)
        kf.landmark[ia[ok]] = ids
        ref.landmark[ib[ok]] = ids
        return int(ok.sum())

    def _local_ba(self):
        """Windowed bundle adjustment over the newest keyframes (planar poses) and the
        landmarks they share. The oldest ``local_ba_fixed`` keyframes are held fixed,
        which also fixes gauge and metric scale inherited from the floor prior."""
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix

        window = self.keyframes[-self.cfg.local_ba_window:]
        if any(kf.pts.shape[0] == 0 for kf in window):
            return None
        free = window[self.cfg.local_ba_fixed:]
        obs_kf, obs_lm, obs_uv = [], [], []
        for i, kf in enumerate(window):
            idx = np.flatnonzero(kf.landmark >= 0)
            lms = kf.landmark[idx]
            keep = self.lm.alive[lms]
            obs_kf.append(np.full(int(keep.sum()), i))
            obs_lm.append(lms[keep])
            obs_uv.append(kf.pts[idx[keep]])
        obs_kf, obs_lm, obs_uv = np.concatenate(obs_kf), np.concatenate(obs_lm), np.concatenate(obs_uv)
        uniq, counts = np.unique(obs_lm, return_counts=True)
        shared = uniq[counts >= 2]
        if len(shared) < 30:
            return None
        sel = np.isin(obs_lm, shared)
        obs_kf, obs_lm, obs_uv = obs_kf[sel], obs_lm[sel], obs_uv[sel]
        lm_col = {lm: j for j, lm in enumerate(shared)}
        obs_j = np.array([lm_col[lm] for lm in obs_lm])
        n_free = len(free)
        fixed_n = self.cfg.local_ba_fixed
        x0 = np.concatenate([np.concatenate([kf.pose for kf in free]), self.lm.pos[shared].ravel()])
        poses_fixed = [kf.pose for kf in window[:fixed_n]]

        def unpack(x):
            poses = poses_fixed + [x[3 * i:3 * i + 3] for i in range(n_free)]
            pts = x[3 * n_free:].reshape(-1, 3)
            return poses, pts

        def residuals(x):
            poses, pts = unpack(x)
            r = np.empty((len(obs_kf), 2))
            for i, pose in enumerate(poses):
                m = obs_kf == i
                if m.any():
                    uv, z = self.model.project_world(pts[obs_j[m]], pose)
                    rr = uv - obs_uv[m]
                    rr[z <= 0.02] = 50.0
                    r[m] = rr
            return r.ravel()

        A = lil_matrix((2 * len(obs_kf), len(x0)), dtype=int)
        rows = np.arange(len(obs_kf))
        for k in range(3):
            A[2 * rows, 3 * n_free + 3 * obs_j + k] = 1
            A[2 * rows + 1, 3 * n_free + 3 * obs_j + k] = 1
        movable = obs_kf >= fixed_n
        for k in range(3):
            cols = 3 * (obs_kf[movable] - fixed_n) + k
            A[2 * rows[movable], cols] = 1
            A[2 * rows[movable] + 1, cols] = 1
        before = residuals(x0)
        try:
            res = least_squares(residuals, x0, jac_sparsity=A, loss="huber", f_scale=2.0, max_nfev=8,
                                x_scale="jac", method="trf")
        except (ValueError, np.linalg.LinAlgError):
            return None
        poses, pts = unpack(res.x)
        # Sanity: accept only if the robust cost fell and no pose jumped implausibly.
        jumps = [np.hypot(*(p[:2] - kf.pose[:2])) for p, kf in zip(poses[fixed_n:], free)]
        if res.cost > 0.5 * np.sum(np.minimum(before ** 2, 4.0)) * 1.0 or max(jumps, default=0) > 0.1:
            return None
        for p, kf in zip(poses[fixed_n:], free):
            kf.pose = np.array([p[0], p[1], wrap(p[2])])
        self.lm.pos[shared] = pts
        last = self.keyframes[-1]
        self.pose = last.pose.copy()
        self.ba_runs = getattr(self, "ba_runs", 0) + 1
        return res.cost

    def _cull(self):
        bad = (self.lm.alive & (self.lm.visible >= 10) & (self.lm.found < 0.15 * self.lm.visible)
               & ~(self.lm.confirmed & (self.lm.found >= 5)))
        self.lm.alive &= ~bad
        if self.lm.alive.sum() > self.cfg.max_landmarks:
            # Capacity limit: drop the least re-observed landmarks that are NOT near
            # the current pose, so the local map the robot is tracking stays intact.
            alive = np.flatnonzero(self.lm.alive)
            far = np.hypot(self.lm.pos[alive, 0] - self.pose[0], self.lm.pos[alive, 1] - self.pose[1]) > 2.5
            cand = alive[far]
            excess = min(len(cand), int(self.lm.alive.sum() - self.cfg.max_landmarks))
            order = np.lexsort((-self.lm.visible[cand], self.lm.found[cand]))
            self.lm.alive[cand[order[:excess]]] = False
        if (~self.lm.alive).sum() > 5000:
            remap = self.lm.compact()
            for kf in self.keyframes:
                valid = kf.landmark >= 0
                kf.landmark[valid] = remap[kf.landmark[valid]]

    # ------------------------------------------------------------ relocalization
    def _local_reacquire(self, pts, desc):
        """Guided matching around the dead-reckoned pose (as in tracking, with wide
        windows): local, so far less prone to aliasing than global PnP."""
        guess = self._dr["pose"]
        alive = np.flatnonzero(self.lm.alive)
        dx, dy = self.lm.pos[alive, 0] - guess[0], self.lm.pos[alive, 1] - guess[1]
        rel = np.arctan2(dy, dx) - guess[2]
        rel = np.arctan2(np.sin(rel), np.cos(rel))
        self._frame_candidates = alive[(np.hypot(dx, dy) < 4.0) & (np.abs(rel) < math.radians(80))]
        self._motion_prior = None
        best = None
        for radius in (self.cfg.wide_radius, 2 * self.cfg.wide_radius):
            lm_idx, kp_idx = self._guided(guess, pts, desc, radius, count_visible=False)
            if len(lm_idx) < self.cfg.reloc_min_inliers:
                continue
            pose, inl, H, err = self._pose_from_matches(guess, lm_idx, kp_idx, pts)
            if H is None or inl.sum() < self.cfg.reloc_min_inliers:
                continue
            best = (pose, int(inl.sum()), H)
            break
        self._frame_candidates = None
        return best

    def _relocalize(self, gray, pts, desc, timestamp):
        pose = None
        if self.cfg.reloc_local_first and self._dr is not None:
            pose = self._local_reacquire(pts, desc)
        if pose is None:
            pose = self.global_localize(pts, desc)
        if pose is None:
            self._reloc_candidates.clear()
            return TrackResult(self.status, None, None, None, 0, 0, False, timestamp, "relocalization_failed")
        estimate, inliers, H = pose
        if self._dr is not None:
            if timestamp - self._dr["since"] > self.cfg.reloc_dr_gate_seconds:
                self._dr = None
            else:
                gate = 0.12 + 0.3 * self._dr["travel"]
                gate_h = 0.12 + 0.25 * self._dr["rot"]
                if (np.hypot(*(estimate[:2] - self._dr["pose"][:2])) > gate
                        or abs(wrap(estimate[2] - self._dr["pose"][2])) > gate_h):
                    self._reloc_candidates.clear()
                    return TrackResult(self.status, None, None, None, inliers, inliers, False, timestamp,
                                       "relocalization_disagrees_with_dead_reckoning")
        if self._reloc_candidates:
            prev = self._reloc_candidates[-1]
            if np.hypot(*(estimate[:2] - prev[:2])) > 0.12 or abs(wrap(estimate[2] - prev[2])) > 0.2:
                self._reloc_candidates.clear()
        self._reloc_candidates.append(estimate)
        if len(self._reloc_candidates) < self.cfg.reloc_confirmations:
            return TrackResult(self.status, None, None, None, inliers, inliers, False, timestamp,
                               "relocalization_candidate")
        self._reloc_candidates.clear()
        self._dr = None
        self.pose, self.status, self.failures = estimate, TRACKING, 0
        self.velocity[:] = 0
        self._set_sigma(H, inliers)
        return TrackResult(TRACKING, estimate.copy(), self.position_sigma, self.heading_sigma, inliers,
                           inliers, False, timestamp, "relocalized")

    def global_localize(self, pts, desc):
        alive = np.flatnonzero(self.lm.alive)
        if len(alive) < 50 or len(pts) < 30:
            return None
        matches = self.matcher.knnMatch(desc, self.lm.desc[alive], k=2)
        good = []
        for m in matches:
            if not m or m[0].distance >= 55:
                continue
            if len(m) < 2 or m[0].distance < 0.8 * m[1].distance:
                good.append(m[0])
            elif np.linalg.norm(self.lm.pos[alive[m[0].trainIdx]] - self.lm.pos[alive[m[1].trainIdx]]) < 0.05:
                good.append(m[0])  # near-duplicate landmarks of one physical point are not ambiguous
        if len(good) < self.cfg.reloc_min_inliers:
            return None
        kp = np.array([m.queryIdx for m in good])
        lm = alive[[m.trainIdx for m in good]]
        # EPnP is degenerate on the (mostly coplanar) floor landmarks; use iterative
        # PnP, falling back to SQPnP. Planarity is re-checked below either way.
        ok = False
        for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP):
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    self.lm.pos[lm].astype(np.float64), pts[kp].astype(np.float64), self.model.K, None,
                    iterationsCount=400, reprojectionError=4.0, confidence=0.995, flags=flag)
            except cv2.error:
                ok = False
            if ok and inl is not None:
                break
        if not ok:
            return None
        if not ok or inl is None or len(inl) < self.cfg.reloc_min_inliers:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4)
        T_cw[:3, :3], T_cw[:3, 3] = R, tvec.ravel()
        T_wb = np.linalg.inv(T_cw) @ self.model.T_cam_base
        # Planar consistency: base_link must be upright at the known height.
        if abs(T_wb[2, 3] - self.model.base_z) > 0.05 or T_wb[2, 2] < math.cos(math.radians(4)):
            return None
        pose0 = np.array([T_wb[0, 3], T_wb[1, 3], math.atan2(T_wb[1, 0], T_wb[0, 0])])
        inl = inl.ravel()
        pose, err, H = self.optimize(pose0, self.lm.pos[lm[inl]], pts[kp[inl]])
        n = int((err < self.cfg.inlier_px).sum())
        if n < self.cfg.reloc_min_inliers:
            return None
        return pose, n, H

    def declare_lost(self, reason="external"):
        self.status = LOST
        self.position_sigma = self.heading_sigma = None
        self.velocity[:] = 0

    # ------------------------------------------------------------ persistence
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        alive = self.lm.alive
        np.savez_compressed(directory / "landmarks.npz", pos=self.lm.pos[alive], desc=self.lm.desc[alive],
                            kind=self.lm.kind[alive], visible=self.lm.visible[alive], found=self.lm.found[alive],
                            confirmed=self.lm.confirmed[alive])
        meta = {
            "schema": "amr_rl.vslam-map.v1",
            "map_version": self.map_version,
            "calibration_version": self.model.version,
            "landmarks": int(alive.sum()),
            "keyframes": [{"id": kf.id, "t": kf.timestamp, "pose": kf.pose.tolist()} for kf in self.keyframes],
        }
        (directory / "map.json").write_text(json.dumps(meta, indent=1))
        return meta

    @classmethod
    def load(cls, directory, model: CameraModel, config=None):
        directory = Path(directory)
        meta = json.loads((directory / "map.json").read_text())
        if meta["calibration_version"] != model.version:
            raise ValueError("Saved map was built with a different camera calibration")
        slam = cls(model, config, map_version=meta["map_version"])
        data = np.load(directory / "landmarks.npz")
        slam.lm.add(data["pos"], data["desc"], 0)
        slam.lm.kind[:] = data["kind"]
        slam.lm.visible[:] = data["visible"]
        slam.lm.found[:] = data["found"]
        slam.lm.confirmed[:] = data["confirmed"]
        slam.status = RELOCALIZING
        for item in meta["keyframes"]:
            slam.keyframes.append(Keyframe(item["id"], item["t"], np.array(item["pose"]),
                                           np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0, int)))
        return slam

    def snapshot(self):
        return {
            "status": self.status,
            "pose": None if self.pose is None or self.status not in (TRACKING, PREDICTED) else [float(v) for v in self.pose],
            "position_sigma": self.position_sigma if self.status in (TRACKING, PREDICTED) else None,
            "heading_sigma": self.heading_sigma if self.status in (TRACKING, PREDICTED) else None,
            "landmarks": len(self.lm),
            "keyframes": len(self.keyframes),
            "map_version": self.map_version,
        }
