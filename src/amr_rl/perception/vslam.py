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
from .pose_graph import PoseGraph, apply_to_pose, apply_transform, between, correction, trajectory_distortion

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
    # Share of the feature budget reserved for the image below the horizon (the floor).
    # 0 = one budget for the whole image (default). In the textured worlds high-contrast
    # posters and patterned objects took the budget: floor features per frame 502 -> 354
    # (docs/results/textured-worlds.md). Evaluated, not yet validated for navigation.
    floor_feature_share: float = 0.0
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
    reloc_confirmations: int = 3
    # Global relocalisation (docs/VSLAM.md). "planar2pt": floor matches lifted to the
    # floor plane give (x, y, theta) from two correspondences (RANSAC survives the
    # low inlier ratios that sank PnP), then the pose must be verified by guided
    # re-matching, be clearly better than any other place, and be confirmed over a
    # change of view that agrees with the robot's own commanded motion. "pnp" is the
    # pre-2026-10 method, kept for A/B and rollback (with confirmations=2, no view
    # change, no probation it reproduces the old behaviour).
    reloc_method: str = "planar2pt"
    reloc_knn: int = 8
    reloc_max_hamming: int = 60
    reloc_distinct_m: float = 0.05     # landmarks closer than this are one physical place
    reloc_ratio: float = 0.85          # best vs best match at a different place
    reloc_hyp_range: float = 1.2       # only near floor points seed hypotheses (IPM range error)
    reloc_iters: int = 600
    reloc_score_px: float = 6.0
    reloc_modes: int = 3
    reloc_verify_min: int = 70         # guided inliers at the verified pose (calibrated: docs/VSLAM.md)
    reloc_explained_min: float = 0.10  # fraction of predicted-visible map cells re-found
    reloc_distinct_ratio: float = 1.5  # best place must beat the next place by this factor
    reloc_min_view_change: float = math.radians(20)
    reloc_view_travel: float = 0.10
    reloc_chain_seconds: float = 4.0   # a confirmation chain expires without a new candidate
    reloc_probation_frames: int = 15   # after acceptance: no map growth; failure -> relocalise
    # Retrieval first: with place descriptors (v2 maps), match against the landmarks of
    # the most similar keyframes before the whole map (higher match precision).
    reloc_retrieval: bool = True
    reloc_retrieval_k: int = 5
    # Pool floor seeds over the last few frames of the relocalisation turn (moved into
    # the current frame by the robot's own commanded motion): far from the map's
    # keyframes a single frame rarely holds two correct floor matches. 0 disables.
    # Off (round 6): on the arena probe 0.3-0.5 m recall 2/8 -> 4/8, all 5 starts, 0
    # false; held out (home_a) one false pose 1.08 m off (docs/results/relocalisation.md).
    reloc_pool_frames: int = 0
    # Loop closure (docs/VSLAM.md): active when a place descriptor is attached
    # (``PlanarVSLAM.place``). Candidates: older keyframes with a similar whole-image
    # descriptor; verification as for relocalisation but only against the
    # candidate's own landmarks; acceptance also needs a plausible correction for the
    # distance travelled and a small trajectory distortion after pose-graph
    # optimisation (ROVER-style); then keyframes, landmarks and pose are corrected.
    loop_closure: bool = True
    loop_min_gap_kf: int = 30
    loop_min_gap_s: float = 30.0
    loop_min_travel: float = 0.6       # m of path since the candidate: a revisit, not continuous tracking
    loop_top_k: int = 3                # most similar candidates checked every keyframe (round 5)
    # ...plus up to loop_uncertain_k more: the most similar candidates whose position
    # relative to the current keyframe is uncertain in the pose graph (sigma >=
    # loop_min_sigma): those are the closures that can correct real drift. Round 5:
    # only ~7 of 94 closures fixed >= 3 cm, most were against recently tied-in regions.
    loop_uncertain_k: int = 0          # evaluated in round 6, not adopted (docs/results/loop-closure.md)
    loop_min_sigma: float = 0.03
    loop_min_similarity: float = 0.55
    loop_window_kf: int = 8            # candidate's landmarks: anchored within +- this many keyframes
    loop_verify_min: int = 50          # calibrated: docs/results/loop-closure.md
    loop_max_correction: float = 0.15  # m, plus loop_correction_per_m x path length since the candidate
    loop_correction_per_m: float = 0.08
    loop_distortion_max: float = 0.10  # m, RMS after rigid alignment (ROVER)
    loop_cooldown_kf: int = 5
    # Temporal consistency (as ORB-SLAM's consecutive-keyframe check): a verified
    # loop closes only when another keyframe within this many keyframes verified the
    # same old region with the same correction of the current pose. A wrong pose from
    # repetitive floor texture does not repeat; a true revisit does.
    loop_confirm_window_kf: int = 6
    loop_confirm_m: float = 0.05
    loop_confirm_rad: float = math.radians(2)
    # Evaluation: detect, verify, confirm and check loops exactly as live, but record
    # the accepted constraints (shadow_edges) instead of correcting the map. The same
    # trajectory can then be scored with and without them (paired; live A/B runs
    # diverge after the first closure).
    loop_shadow: bool = False
    # Covisibility edges (ORB-SLAM's essential graph): a keyframe tracked against
    # landmarks created by an older keyframe is already anchored to it. Without these
    # edges the pose graph treats consecutive keyframes as a free odometry chain and
    # drags keyframes that had re-anchored to the old map along with a loop correction
    # (shadow evaluation: correct loops made 6/12 maps worse, e.g. 5.8 -> 23 cm).
    # Offline they reduced that to 4/12 but did not make loop correction a net gain;
    # not validated live, so off (covis_max_per_kf = 3 to enable).
    covis_min_shared: int = 30
    covis_max_per_kf: int = 0
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
    # In-place turns (round 7; docs/results/turn-drift.md). Heading drifted up to 28 deg
    # per turn near objects: features on object faces below the camera are lifted to
    # the floor too far away, and the camera's 0.125 m lever arm turns that depth
    # error into extra yaw; the error then locks in through landmarks made during the
    # turn. A turn is recognised from the robot's own command (engineered rule):
    # |v| < turn_v_max and |w| >= turn_w_min for 2 frames; pauses up to turn_pause_s.
    turn_w_min: float = 0.15
    turn_v_max: float = 0.02
    turn_pause_s: float = 1.0
    # C7: down-weight matches by how much a landmark depth error moves them during a
    # rotation (sigma ~ f * lever * angle * rel_depth_error / depth). off|turn|always
    depth_weighting: str = "off"
    depth_rel_ipm: float = 0.5
    depth_rel_tri: float = 0.05
    depth_ref_angle: float = 0.35
    # C1: estimate heading only during a turn, the base pinned at the turn axis
    # ("hard") or held by a tight prior ("prior"); falls back to the full fit when
    # pinning fits clearly worse (the base really moved). off|hard|prior
    turn_pin_xy: str = "off"
    turn_xy_sigma: float = 0.002
    turn_fallback_ratio: float = 1.5
    # C3: landmarks made during a turn: "gaps" only in image cells without matched
    # landmarks; "tentative" never confirmed until the base moves 5 cm (or validated).
    turn_landmarks: str = "normal"
    turn_gap_cell: int = 40
    turn_gap_min: int = 2
    # C2a: validate floor (IPM) landmarks by triangulation between the keyframe that
    # made them and a later one; non-floor points become triangulated landmarks.
    # Until validated they are not confirmed by the 4 cm rule. off|turn|all
    floor_validation: str = "off"
    floor_val_min_parallax_deg: float = 2.0
    floor_val_height: float = 0.03
    floor_val_ratio: float = 0.12  # |triangulated / floor-lifted distance - 1| <= this: on the floor
    # C4: the turn the robot measures must agree with the turn it commanded (wheel
    # slip only makes the real turn smaller): flag over-rotation and angular freezes.
    turn_cmd_check: str = "off"   # off|flag
    turn_k_max: float = 1.0
    turn_k_min: float = 0.6
    turn_cmd_margin: float = 0.10
    turn_sigma_growth: float = 0.03
    # C5: turn closure: on returning to the turn's start heading (or after each full
    # revolution), re-find the heading against landmarks that existed before the turn
    # and spread the correction over the turn's keyframes. off|rotate|purge
    turn_closure: str = "off"
    closure_window_deg: float = 40.0
    closure_search_deg: float = 24.0
    closure_step_deg: float = 3.0
    closure_min_inliers: int = 50
    closure_ratio: float = 1.5
    closure_max_frac: float = 0.08
    closure_confirm_deg: float = 1.0
    # Side fix: count a landmark as "visible" once per frame (not once per hypothesis
    # and search radius), so the cull rule sees the true re-find rate.
    visible_once_per_frame: bool = False


@dataclass
class Keyframe:
    id: int
    timestamp: float
    pose: np.ndarray
    pts: np.ndarray
    desc: np.ndarray
    landmark: np.ndarray  # landmark index per feature or -1
    gray: np.ndarray | None = None
    gdesc: np.ndarray | None = None  # whole-image place descriptor (loop closure / retrieval)


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
        self.anchor = np.zeros(0, int)  # keyframe that created it (-1: unknown); moves with it on loop closure
        self.flags = np.zeros(0, np.int8)  # bit0: made during an in-place turn; bit1: validated floor point

    def __len__(self):
        return int(self.alive.sum())

    def add(self, pos, desc, kind, origin=None, confirmed=False, created=-np.inf, anchor=-1, flags=0):
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
        self.anchor = np.concatenate([self.anchor, np.full(n, int(anchor))])
        self.flags = np.concatenate([self.flags, np.full(n, int(flags), np.int8)])
        return np.arange(start, start + n)

    def compact(self):
        keep = self.alive
        remap = -np.ones(len(keep), int)
        remap[keep] = np.arange(int(keep.sum()))
        for name in ("pos", "desc", "kind", "visible", "found", "alive", "confirmed", "origin", "marked", "created",
                     "anchor", "flags"):
            setattr(self, name, getattr(self, name)[keep])
        return remap


class PlanarVSLAM:
    def __init__(self, model: CameraModel, config: VSLAMConfig | None = None, *, map_version=None):
        self.model = model
        self.cfg = config or VSLAMConfig()
        self.orb = cv2.ORB_create(nfeatures=self.cfg.n_features, scaleFactor=1.2, nlevels=6,
                                  edgeThreshold=15, patchSize=15, fastThreshold=self.cfg.fast_threshold)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._split = None
        if self.cfg.floor_feature_share > 0:
            n_floor = int(round(self.cfg.n_features * self.cfg.floor_feature_share))
            kw = dict(scaleFactor=1.2, nlevels=6, edgeThreshold=15, patchSize=15, fastThreshold=self.cfg.fast_threshold)
            # horizon row: a floor point far ahead (8 m) projects just below it
            uv, _ = model.project_world(np.array([[8.0, 0.0, 0.0]]), (0.0, 0.0, 0.0))
            row = int(np.clip(uv[0, 1], 1, model.height - 2))
            masks = np.zeros((2, model.height, model.width), np.uint8)
            masks[0, row:] = 255
            masks[1, :row] = 255
            self._split = [(cv2.ORB_create(nfeatures=n_floor, **kw), masks[0]),
                           (cv2.ORB_create(nfeatures=self.cfg.n_features - n_floor, **kw), masks[1])]
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
        self._reloc_candidates: list = []  # (pose, odometry pose at that frame)
        self._reloc_odom = np.zeros(3)  # integrated commanded motion while relocalising
        self._reloc_info = {}
        self._probation = 0
        self._probation_dr = None
        self.place = None  # whole-image place descriptor (perception/place_recognition.py) or None
        self.loop_log = []  # every loop candidate checked (operational values; bounded)
        self.loop_edges = []  # accepted loop constraints: (i, j, z)
        self.on_loop_closure = None  # callback(corrections: {keyframe id: planar transform}) for map layers
        self._loop_cooldown = 0
        self._loop_pending = []  # verified loop hypotheses awaiting a consistent second one
        self._reloc_pool = []  # [(odometry pose, floor points (n,2) robot frame, landmark ids, ranges)]
        self.shadow_edges = []  # loop_shadow: accepted constraints not applied (i, j, z)
        self.covis_edges = []  # (older anchor keyframe, keyframe, z, shared landmarks)
        self.n_loaded_keyframes = 0  # keyframes of a loaded map stay fixed in the pose graph
        self.horizon = model.horizon_row()
        self.frames = 0
        self._turn = None  # in-place turn state (see _update_turn_state)
        self._turn_count = 0
        self._visible_frame = -1
        self.turn_closures = []  # applied turn closures (operational log)
        self._lever = float(np.linalg.norm(model.T_base_cam[:2, 3]))
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
        self.reloc_log = []  # every relocalisation outcome (operational values only; bounded)
        self._last_err = None
        self._last_quality = (0, 0.0)
        self.distance_travelled = 0.0

    # ------------------------------------------------------------ features
    def features(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = self.clahe.apply(gray)
        if self._split is not None and gray.shape == self._split[0][1].shape:
            kps, descs = [], []
            for orb, mask in self._split:
                k, d = orb.detectAndCompute(gray, mask)
                if d is not None and k:
                    kps += list(k)
                    descs.append(d)
            desc = np.vstack(descs) if descs else None
        else:
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

    def optimize(self, pose0, P, uv, iters=10, weights=None, prior=None, pin=None):
        """Robust planar pose fit. ``prior`` = (pose, sigmas): the robot's own commanded
        motion as a weak Gaussian prior (whitened, Huber-robust); vision dominates
        whenever it is informative. ``pin`` = (mode, xy): during an in-place turn the
        base stays at xy ("hard": heading only) or is held there by a tight
        un-robustified prior ("prior", sigma turn_xy_sigma)."""
        x = np.asarray(pose0, float).copy()
        if pin is not None and pin[0] == "hard":
            x[:2] = pin[1]
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
            if pin is not None and pin[0] == "prior":
                sp = self.cfg.turn_xy_sigma
                H = H + np.diag([1 / sp ** 2, 1 / sp ** 2, 0.0])
                g = g + np.array([(x[0] - pin[1][0]) / sp ** 2, (x[1] - pin[1][1]) / sp ** 2, 0.0])
            try:
                if pin is not None and pin[0] == "hard":
                    dx = np.array([0.0, 0.0, -g[2] / (H[2, 2] + 1e-6)])
                else:
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
            if self.cfg.visible_once_per_frame:
                if self._visible_frame != self.frames or len(self._visible_seen) != len(self.lm.pos):
                    self._visible_frame, self._visible_seen = self.frames, np.zeros(len(self.lm.pos), bool)
                new = cand[~self._visible_seen[cand]]
                self.lm.visible[new] += 1
                self._visible_seen[new] = True
            else:
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
        tr = self._turn
        if tr is None or self.cfg.turn_pin_xy == "off":
            return self._fit(pose0, lm_idx, kp_idx, pts)
        pinned = self._fit(pose0, lm_idx, kp_idx, pts, pin=(self.cfg.turn_pin_xy, tr["axis"]))
        free = self._fit(pose0, lm_idx, kp_idx, pts)
        if pinned[2] is None:
            return free
        if free[2] is not None:
            e_p = float(np.mean(pinned[3][pinned[1]])) if pinned[1].any() else np.inf
            e_f = float(np.mean(free[3][free[1]])) if free[1].any() else np.inf
            if e_p > self.cfg.turn_fallback_ratio * max(e_f, 0.3) or pinned[1].sum() < 0.75 * free[1].sum():
                tr["fallbacks"] += 1  # the base really moved (slide, push): follow it
                tr["axis"] = free[0][:2].copy()
                return free
        return pinned

    def _depth_weights(self, lm_idx, pose):
        """C7: how far a landmark's image position moves, for a rotation of
        depth_ref_angle, per its relative depth error (lever arm a, depth d):
        sigma_px = f * a * angle * rel / d; weight = huber^2 / (huber^2 + sigma^2)."""
        mode = self.cfg.depth_weighting
        if mode == "off" or (mode == "turn" and self._turn is None):
            return None
        cam = self.model.T_world_cam(pose)[:3, 3]
        d = np.maximum(np.linalg.norm(self.lm.pos[lm_idx] - cam, axis=1), 0.05)
        exact = (self.lm.kind[lm_idx] == 1) | ((self.lm.flags[lm_idx] & 2) != 0)
        rel = np.where(exact, self.cfg.depth_rel_tri, self.cfg.depth_rel_ipm)
        sig = self.model.K[0, 0] * self._lever * self.cfg.depth_ref_angle * rel / d
        hub = self.cfg.huber_px ** 2
        return hub / (hub + sig ** 2)

    def _fit(self, pose0, lm_idx, kp_idx, pts, pin=None):
        P, uv = self.lm.pos[lm_idx], pts[kp_idx]
        prior = self._motion_prior
        dw = self._depth_weights(lm_idx, pose0)
        pose, err, _ = self.optimize(pose0, P, uv, prior=prior, weights=dw, pin=pin)
        inl = err < self.cfg.inlier_px * 2
        if inl.sum() >= self.cfg.min_inliers:
            use = inl & self.lm.confirmed[lm_idx]
            if use.sum() < self.cfg.min_inliers:
                use = inl  # early map: tentative floor points are all there is
            # Established landmarks (re-observed many times) anchor the estimate more
            # strongly than fresh ones, limiting drift from newly added points.
            weights = np.clip(0.3 + self.lm.found[lm_idx[use]] / 8.0, 0.3, 1.5)
            if dw is not None:
                weights = weights * dw[use]
            pose, err, H = self.optimize(pose, P[use], uv[use], weights=weights, prior=prior, pin=pin)
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
        if self._turn is not None and self.cfg.turn_pin_xy != "off":
            for _, h in out:
                h[:2] = self._turn["axis"]
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
            self._turn, self._turn_count = None, 0
            return self._relocalize(gray, pts, desc, timestamp, commanded=commanded, dt=dt, rgb=rgb)
        # TRACKING or PREDICTED: try to (re)acquire against the local map.
        self._update_turn_state(commanded, dt, timestamp)
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
        if result is None and self._probation > 0:
            return self._reject_in_probation(timestamp, "relocalization_rejected_in_probation")
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
                self._turn, self._turn_count = None, 0
                return TrackResult(LOST, None, None, None, 0, 0, False, timestamp, "tracking_failed")
            self.status = PREDICTED
            return TrackResult(PREDICTED, self.pose.copy(), self.position_sigma, self.heading_sigma,
                               0, 0, False, timestamp, "visual_tracking_interrupted", info={"degraded": True})
        self.status = TRACKING
        self.predicted_time = 0.0
        pose, inl, H, lm_idx, kp_idx, err = result
        self._frame_candidates = None
        if self._probation > 0 and not self._motion_consistent(pose, commanded, dt, timestamp):
            return self._reject_in_probation(timestamp, "relocalization_rejected_in_probation")
        if self._probation == 0 and not self._motion_consistent(pose, commanded, dt, timestamp):
            self._on_frozen(timestamp)
            self.status = LOST
            self.position_sigma = self.heading_sigma = None
            self.velocity[:] = 0
            self._motion_log.clear()
            self._turn, self._turn_count = None, 0
            return TrackResult(LOST, None, None, None, 0, 0, False, timestamp, "visual_motion_inconsistent_with_commands")
        self.failures = 0
        if dt > 0:
            step = np.array([pose[0] - self.pose[0], pose[1] - self.pose[1], wrap(pose[2] - self.pose[2])])
            self.velocity = 0.5 * self.velocity + 0.5 * step / dt
            self.distance_travelled += float(np.hypot(*step[:2]))
        self.pose = pose
        if self._probation > 0:
            # A just-accepted relocalisation is not trusted to change the map until it
            # has tracked for a while (a wrong pose would otherwise extend the map
            # from itself and then track its own landmarks).
            self._probation -= 1
            self._last_quality = (int(inl.sum()), float(np.mean(err[inl])))
            self._set_sigma(H, int(inl.sum()))
            return TrackResult(TRACKING, pose.copy(), self.position_sigma, self.heading_sigma,
                               int(inl.sum()), len(lm_idx), False, timestamp, "probation",
                               inlier_uv=pts[kp_idx[inl]])
        self.lm.found[lm_idx[inl]] += 1
        centre = self.model.T_world_cam(pose)[:3, 3]
        tent = lm_idx[inl][~self.lm.confirmed[lm_idx[inl]]]
        if len(tent):
            moved = np.linalg.norm(self.lm.origin[tent] - centre, axis=1) >= 0.04
            ok = moved & (self.lm.found[tent] >= 3)
            ok &= ~self._unconfirmable(tent, centre)
            self.lm.confirmed[tent[ok]] = True
        self._last_quality = (int(inl.sum()), float(np.mean(err[inl])))
        self._set_sigma(H, int(inl.sum()))
        info = {}
        tr = self._turn
        if tr is not None:
            tr["est_rot"] += wrap(pose[2] - tr["last_th"])
            tr["last_th"] = float(pose[2])
            tr["max_exc"] = max(tr["max_exc"], abs(tr["est_rot"] - tr["rot_at_closure"]))
            if self.cfg.turn_pin_xy != "off":
                self.position_sigma = max(self.position_sigma, 0.02)  # measured base slide in turns: 1-3 cm
            closure = self._turn_closure_check(pts, desc, timestamp) if self.cfg.turn_closure != "off" else None
            if closure is not None:
                info["turn_closure"] = closure
            if self.cfg.turn_cmd_check != "off":
                self._turn_command_check(tr)
                if self.cfg.turn_sigma_growth > 0:
                    self.heading_sigma = float(np.hypot(self.heading_sigma, self.cfg.turn_sigma_growth
                                                        * abs(tr["est_rot"] - tr["rot_at_closure"])))
            info.update({"turn": True, "turn_rot": float(tr["est_rot"]), "degraded": bool(tr["flagged"])})
        matched_lm = -np.ones(len(pts), int)
        matched_lm[kp_idx[inl]] = lm_idx[inl]
        kf = False
        if not info.get("degraded"):  # never extend the map from a flagged turn
            kf = self._maybe_keyframe(gray, pts, desc, matched_lm, int(inl.sum()), timestamp)
        if kf and self.place is not None and self.cfg.loop_closure:
            self._loop_check(rgb, timestamp)
        return TrackResult(TRACKING, self.pose.copy(), self.position_sigma, self.heading_sigma,
                           int(inl.sum()), len(lm_idx), kf, timestamp,
                           inlier_uv=pts[kp_idx[inl]], info=info)

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
        ids = self.lm.add(world[ok], desc[ok], 0, origin=self.model.T_world_cam(pose)[:3, 3], created=timestamp,
                          anchor=0)
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
        if self.cfg.floor_validation != "off":
            self._validate_floor(kf)
        # Remaining unmatched floor-like features become metric floor landmarks.
        free = kf.landmark < 0
        world, ok = self._ipm(pts, self.pose)
        ok &= free
        if self._turn is not None and self.cfg.turn_landmarks == "gaps" and len(pts):
            cell = np.floor(pts / self.cfg.turn_gap_cell).astype(int)
            key = cell[:, 0] * 1000 + cell[:, 1]
            matched_keys, counts = np.unique(key[matched_lm >= 0], return_counts=True)
            covered = matched_keys[counts >= self.cfg.turn_gap_min]
            ok &= ~np.isin(key, covered)  # only map views the existing map does not cover
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
                          created=self._now, anchor=kf.id, flags=1 if self._turn is not None else 0)
        kf.landmark[np.flatnonzero(ok)] = ids
        self._note_covisibility(kf, matched_lm)
        self.keyframes.append(kf)
        if self._turn is not None:
            self._turn["kf_rot"][kf.id] = float(self._turn["est_rot"])
        pinned = self._turn is not None and self.cfg.turn_pin_xy != "off"
        if self.cfg.local_ba_window >= 4 and len(self.keyframes) >= self.cfg.local_ba_window and not pinned:
            self._local_ba()
        if len(self.keyframes) > 400:  # keep images only for recent keyframes
            self.keyframes[-400].gray = None
        self._cull()
        return True

    # ------------------------------------------------------------ in-place turns
    def _update_turn_state(self, commanded, dt, t):
        """Recognise an in-place turn from the robot's own command (engineered rule)."""
        cfg = self.cfg
        if commanded is None or self.pose is None:
            self._turn, self._turn_count = None, 0
            return
        v, w = commanded
        turning = abs(v) < cfg.turn_v_max and abs(w) >= cfg.turn_w_min
        tr = self._turn
        if tr is not None:
            if abs(v) >= cfg.turn_v_max or (not turning and t - tr["last_turning"] > cfg.turn_pause_s):
                self._turn, self._turn_count = None, 0
            else:
                if turning:
                    tr["last_turning"] = t
                tr["cmd_rot"] += w * dt
                return
        self._turn_count = self._turn_count + 1 if turning else 0
        if turning and self._turn_count >= 2:
            self._turn = {"axis": self.pose[:2].copy(), "theta0": float(self.pose[2]), "t0": t, "last_turning": t,
                          "cmd_rot": w * dt, "est_rot": 0.0, "last_th": float(self.pose[2]), "max_exc": 0.0,
                          "rot_at_closure": 0.0, "kf_rot": {}, "flagged": False, "pending": None, "closures": 0,
                          "fallbacks": 0}

    def _unconfirmable(self, tent, centre):
        """Tentative landmarks that the 4 cm rule must not confirm (C3 tentative, C2a)."""
        out = np.zeros(len(tent), bool)
        born_in_turn = (self.lm.flags[tent] & 1) != 0
        if self.cfg.turn_landmarks == "tentative":
            far = np.linalg.norm(self.lm.origin[tent] - centre, axis=1) >= 0.10
            out |= born_in_turn & ((self._turn is not None) | ~far)
        if self.cfg.floor_validation != "off":
            scope = born_in_turn if self.cfg.floor_validation == "turn" else np.ones(len(tent), bool)
            out |= scope & (self.lm.kind[tent] == 0) & ((self.lm.flags[tent] & 2) == 0)
        return out

    def _validate_floor(self, kf):
        """C2a: triangulate tentative floor landmarks seen in ``kf`` against the keyframe
        that created them. On the floor -> validated and confirmed; above it -> a
        triangulated landmark at the measured position; below it -> removed."""
        cfg = self.cfg
        ids = kf.landmark[kf.landmark >= 0]
        if len(ids) == 0:
            return 0
        lm = self.lm
        sel = (lm.kind[ids] == 0) & ~lm.confirmed[ids] & ((lm.flags[ids] & 2) == 0) & (lm.anchor[ids] >= 0) \
            & (lm.anchor[ids] != kf.id)
        if cfg.floor_validation == "turn":
            sel &= (lm.flags[ids] & 1) != 0
        ids = ids[sel]
        if len(ids) == 0:
            return 0
        by = {k.id: k for k in self.keyframes}
        where = {int(lid): i for i, lid in enumerate(kf.landmark) if lid >= 0}
        done = 0
        for a in np.unique(lm.anchor[ids]):
            A = by.get(int(a))
            if A is None or A.pts is None or len(A.pts) == 0:
                continue
            in_a = {int(lid): j for j, lid in enumerate(A.landmark) if lid >= 0}
            group = [int(i) for i in ids[lm.anchor[ids] == a] if int(i) in in_a]
            if not group:
                continue
            ka = np.array([where[i] for i in group])
            ja = np.array([in_a[i] for i in group])
            T1 = np.linalg.inv(self.model.T_world_cam(kf.pose))
            T2 = np.linalg.inv(self.model.T_world_cam(A.pose))
            K = self.model.K
            X = cv2.triangulatePoints(K @ T1[:3], K @ T2[:3], kf.pts[ka].T.astype(float), A.pts[ja].T.astype(float))
            X = (X[:3] / X[3]).T
            uv1, z1 = self._project(X, kf.pose)
            uv2, z2 = self._project(X, A.pose)
            e = np.maximum(np.linalg.norm(uv1 - kf.pts[ka], axis=1), np.linalg.norm(uv2 - A.pts[ja], axis=1))
            c1, c2 = np.linalg.inv(T1)[:3, 3], np.linalg.inv(T2)[:3, 3]
            r1, r2 = X - c1, X - c2
            cos = (r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-9)
            par = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            good = (z1 > 0.05) & (z2 > 0.05) & (e < 2.0) & (par >= cfg.floor_val_min_parallax_deg)
            g = np.array(group)
            # Distance along the ray: triangulated vs where the floor assumption put it.
            # A point on an object face is much nearer than its floor lifting; a floor
            # point agrees within the triangulation noise (a fixed height threshold
            # misread noisy floor points as objects).
            ratio = np.linalg.norm(r1, axis=1) / np.maximum(np.linalg.norm(lm.pos[g] - c1, axis=1), 1e-6)
            on_floor = good & (np.abs(ratio - 1.0) <= cfg.floor_val_ratio)
            above = good & (ratio < 1.0 - 2 * cfg.floor_val_ratio) & (X[:, 2] > cfg.floor_val_height) & (X[:, 2] < 2.0)
            below = good & (ratio > 1.0 + 2 * cfg.floor_val_ratio) & (X[:, 2] < -cfg.floor_val_height)
            lm.flags[g[on_floor]] |= 2
            lm.confirmed[g[on_floor]] = True
            lm.pos[g[above]] = X[above]
            lm.kind[g[above]] = 1
            lm.flags[g[above]] |= 2
            lm.confirmed[g[above]] = True
            lm.alive[g[below]] = False
            done += int(good.sum())
        return done

    def _turn_command_check(self, tr):
        """C4: the measured turn must not exceed the commanded one (slip only reduces
        it) and must not stay far below it (angular freeze)."""
        cfg = self.cfg
        cmd, est = abs(tr["cmd_rot"]), abs(tr["est_rot"])
        over = est > cfg.turn_k_max * cmd + cfg.turn_cmd_margin + 0.05 * cmd
        frozen = cmd >= 0.5 and est < cfg.turn_k_min * cmd - cfg.turn_cmd_margin
        if over or frozen:
            tr["flagged"] = True

    def _turn_closure_check(self, pts, desc, t):
        """C5: back at the turn's start heading (after >= 60 deg away), re-find the
        heading against landmarks that existed before the turn; two consecutive
        agreeing frames, a distinct best mode and a bounded correction are required."""
        cfg, tr = self.cfg, self._turn
        if tr["max_exc"] < math.radians(60) or abs(wrap(self.pose[2] - tr["theta0"])) > math.radians(cfg.closure_window_deg):
            return None
        anchors = np.flatnonzero(self.lm.alive & (self.lm.created < tr["t0"]))
        if len(anchors) < cfg.closure_min_inliers or len(pts) < cfg.closure_min_inliers:
            return None
        saved = self._frame_candidates
        self._frame_candidates = anchors
        xy = tr["axis"] if cfg.turn_pin_xy != "off" else self.pose[:2]
        n = int(round(cfg.closure_search_deg / cfg.closure_step_deg))
        found = []
        for j in range(-n, n + 1):
            cand = np.array([xy[0], xy[1], wrap(self.pose[2] + math.radians(j * cfg.closure_step_deg))])
            li, ki = self._guided(cand, pts, desc, 12.0, count_visible=False)
            if len(li) < cfg.closure_min_inliers // 2:
                continue
            pose, err, _ = self.optimize(cand, self.lm.pos[li], pts[ki], pin=("hard", xy))
            inl = err < self.cfg.inlier_px
            if inl.sum():
                found.append((int(inl.sum()), float(np.mean(err[inl])), float(pose[2])))
        self._frame_candidates = saved
        entry = {"t": float(t), "candidates": len(found)}
        if not found:
            return None
        found.sort(key=lambda f: -f[0])
        best = found[0]
        rivals = [f[0] for f in found[1:] if abs(wrap(f[2] - best[2])) > math.radians(6)]
        second = max(rivals, default=0)
        delta = wrap(best[2] - self.pose[2])
        entry.update({"inliers": best[0], "mean_px": best[1], "second": second, "delta_deg": math.degrees(delta)})
        if best[0] < cfg.closure_min_inliers or best[1] >= 2.0 or best[0] < cfg.closure_ratio * second:
            entry["outcome"] = "rejected_weak"
            return None
        turned = abs(tr["est_rot"] - tr["rot_at_closure"])
        if abs(delta) > max(math.radians(4), cfg.closure_max_frac * turned):
            tr["pending"] = None
            entry["outcome"] = "rejected_too_large"
            self.turn_closures.append(entry)
            return None
        if abs(delta) < math.radians(0.2):
            tr["max_exc"], tr["rot_at_closure"], tr["pending"] = 0.0, tr["est_rot"], None
            return None  # nothing to correct
        if tr["pending"] is None or abs(wrap(delta - tr["pending"])) > math.radians(cfg.closure_confirm_deg):
            tr["pending"] = delta
            return None
        self._apply_turn_closure(delta, xy)
        tr["max_exc"], tr["rot_at_closure"], tr["pending"] = 0.0, tr["est_rot"], None
        tr["closures"] += 1
        entry["outcome"] = "closed"
        self.turn_closures.append(entry)
        return entry

    def _apply_turn_closure(self, delta, xy):
        """Rotate the turn's keyframes about the axis by their share of the turn
        (purge mode: drop their landmarks instead of moving them), and the pose."""
        tr = self._turn
        total = max(abs(tr["est_rot"] - tr["rot_at_closure"]), 1e-6)
        by = {k.id: k for k in self.keyframes}
        corrections = {}
        for kid, rot in tr["kf_rot"].items():
            k = by.get(kid)
            if k is None or kid < self.n_loaded_keyframes:
                continue
            share = float(np.clip(abs(rot - tr["rot_at_closure"]) / total, 0.0, 1.0))
            d = delta * share
            c, s_ = math.cos(d), math.sin(d)
            before = k.pose.copy()
            rel = k.pose[:2] - xy
            k.pose = np.array([xy[0] + c * rel[0] - s_ * rel[1], xy[1] + s_ * rel[0] + c * rel[1], wrap(k.pose[2] + d)])
            T = correction(before, k.pose)
            corrections[kid] = T
            sel = self.lm.anchor == kid
            if sel.any():
                if self.cfg.turn_closure == "purge":
                    self.lm.alive[sel] = False
                else:
                    self.lm.pos[sel, :2] = apply_transform(T, self.lm.pos[sel, :2])
        c, s_ = math.cos(delta), math.sin(delta)
        rel = self.pose[:2] - xy
        self.pose = np.array([xy[0] + c * rel[0] - s_ * rel[1], xy[1] + s_ * rel[0] + c * rel[1], wrap(self.pose[2] + delta)])
        tr["est_rot"] += delta
        tr["last_th"] = float(self.pose[2])
        self.velocity[:] = 0
        self._motion_log.clear()  # the jump is a correction, not motion
        self._kp_tree = self._kp_tree_pts = None
        if corrections and self.on_loop_closure is not None:
            self.on_loop_closure(corrections, {k.id: k.timestamp for k in self.keyframes})

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
                          created=self._now, anchor=kf.id)
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

    def _relocalize(self, gray, pts, desc, timestamp, commanded=None, dt=0.0, rgb=None):
        v, w = commanded if commanded is not None else (0.0, 0.0)
        if dt > 0:  # odometry of the robot's own commands, to chain confirmations
            th = self._reloc_odom[2] + 0.5 * w * dt
            self._reloc_odom = self._reloc_odom + np.array([v * dt * math.cos(th), v * dt * math.sin(th), w * dt])
        pose = None
        if self.cfg.reloc_local_first and self._dr is not None:
            pose = self._local_reacquire(pts, desc)
        if pose is None:
            pose = self.global_localize(pts, desc, rgb=rgb, pool=True)
        info = dict(self._reloc_info)
        if pose is None:
            if self.cfg.reloc_method == "pnp":
                self._reloc_candidates.clear()
            self._expire_chain(timestamp)
            self._log_reloc(timestamp, "relocalization_failed", **info)
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
                    self._log_reloc(timestamp, "relocalization_disagrees_with_dead_reckoning", inliers, estimate, **info)
                    return TrackResult(self.status, None, None, None, inliers, inliers, False, timestamp,
                                       "relocalization_disagrees_with_dead_reckoning")
        if not self._chain_accepts(estimate, timestamp):
            self._reloc_candidates.clear()
        self._reloc_candidates.append((estimate, self._reloc_odom.copy(), timestamp))
        if not self._chain_complete():
            self._log_reloc(timestamp, "relocalization_candidate", inliers, estimate, **info)
            return TrackResult(self.status, None, None, None, inliers, inliers, False, timestamp,
                               "relocalization_candidate")
        self._reloc_candidates.clear()
        self._reloc_pool.clear()
        self._probation_dr = self._dr
        self._dr = None
        self._probation = self.cfg.reloc_probation_frames
        self.pose, self.status, self.failures = estimate, TRACKING, 0
        self.velocity[:] = 0
        self._motion_log.clear()
        self._set_sigma(H, inliers)
        self._log_reloc(timestamp, "relocalized", inliers, estimate, **info)
        return TrackResult(TRACKING, estimate.copy(), self.position_sigma, self.heading_sigma, inliers,
                           inliers, False, timestamp, "relocalized")

    def _relative_odom(self, a, b):
        """Commanded motion from odometry pose a to b, expressed in a's frame."""
        d = b[:2] - a[:2]
        c, s_ = math.cos(a[2]), math.sin(a[2])
        return np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1], wrap(b[2] - a[2])])

    def _chain_accepts(self, estimate, t):
        """Does a new candidate continue the confirmation chain? Legacy: within 0.12 m /
        0.2 rad of the previous candidate. Planar: the previous candidate moved by the
        commanded motion since must land on this one."""
        if not self._reloc_candidates:
            return True
        prev, odom_prev, t_prev = self._reloc_candidates[-1]
        if self.cfg.reloc_method == "pnp":
            return np.hypot(*(estimate[:2] - prev[:2])) <= 0.12 and abs(wrap(estimate[2] - prev[2])) <= 0.2
        if t - t_prev > self.cfg.reloc_chain_seconds:
            return False
        rel = self._relative_odom(odom_prev, self._reloc_odom)
        c, s_ = math.cos(prev[2]), math.sin(prev[2])
        pred = np.array([prev[0] + c * rel[0] - s_ * rel[1], prev[1] + s_ * rel[0] + c * rel[1], prev[2] + rel[2]])
        travel = float(np.hypot(*rel[:2]))
        return (np.hypot(*(estimate[:2] - pred[:2])) <= 0.06 + 0.3 * travel
                and abs(wrap(estimate[2] - pred[2])) <= 0.08 + 0.2 * abs(rel[2]))

    def _chain_complete(self):
        if len(self._reloc_candidates) < self.cfg.reloc_confirmations:
            return False
        first, last = self._reloc_candidates[0][1], self._reloc_candidates[-1][1]
        rel = self._relative_odom(first, last)
        # The candidates must come from clearly different views: two near-identical
        # frames confirm an aliased match as readily as a true one.
        rot = sum(abs(wrap(b[1][2] - a[1][2])) for a, b in zip(self._reloc_candidates[:-1], self._reloc_candidates[1:]))
        return (self.cfg.reloc_min_view_change <= 0 or rot >= self.cfg.reloc_min_view_change
                or float(np.hypot(*rel[:2])) >= self.cfg.reloc_view_travel)

    def _expire_chain(self, t):
        if self._reloc_candidates and t - self._reloc_candidates[-1][2] > self.cfg.reloc_chain_seconds:
            self._reloc_candidates.clear()

    def _reject_in_probation(self, timestamp, reason):
        self._probation = 0
        self._dr = self._probation_dr
        self.status = RELOCALIZING
        self.position_sigma = self.heading_sigma = None
        self.velocity[:] = 0
        self._motion_log.clear()
        self._log_reloc(timestamp, reason)
        return TrackResult(RELOCALIZING, None, None, None, 0, 0, False, timestamp, reason)

    def _log_reloc(self, t, outcome, inliers=0, pose=None, **extra):
        self.reloc_log.append({"t": float(t), "outcome": outcome, "inliers": int(inliers),
                               "pose": None if pose is None else [float(v) for v in pose], **extra})
        del self.reloc_log[:-2000]

    def global_localize(self, pts, desc, rgb=None, pool=False):
        """Pose from the map, or None. Returns (pose, inliers, H). With a place
        descriptor and described keyframes, the landmarks of the most similar
        keyframes are tried first, then the whole map."""
        self._reloc_info = {}
        if self.cfg.reloc_method == "pnp":
            return self._global_localize_pnp(pts, desc)
        subset = self._retrieved_landmarks(rgb)
        if subset is not None:
            out = self._global_localize_planar(pts, desc, candidates=subset)
            if out is not None:
                self._reloc_info["retrieval"] = True
                return out
        return self._global_localize_planar(pts, desc, pool=pool)

    def _retrieved_landmarks(self, rgb):
        if (rgb is None or self.place is None or not self.cfg.reloc_retrieval
                or not any(k.gdesc is not None for k in self.keyframes)):
            return None
        q = np.asarray(self.place.describe(rgb), np.float32)
        kfs = [k for k in self.keyframes if k.gdesc is not None]
        sims = np.array([float(q @ k.gdesc) for k in kfs])
        top = [kfs[i].id for i in np.argsort(-sims)[:self.cfg.reloc_retrieval_k] if sims[i] >= self.cfg.loop_min_similarity]
        if not top:
            return None
        w = self.cfg.loop_window_kf
        sel = self.lm.alive & (self.lm.anchor >= 0) & np.any(
            np.abs(self.lm.anchor[:, None] - np.array(top)[None]) <= w, axis=1)
        idx = np.flatnonzero(sel)
        return idx if len(idx) >= 50 else None

    def _global_localize_pnp(self, pts, desc):
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

    # ------------------------------------------------------------ loop closure
    def _loop_check(self, rgb, timestamp):
        """Describe the new keyframe; look for an earlier visit of the same place and,
        if a candidate passes verification and the pose-graph checks, correct the map."""
        cfg = self.cfg
        kf = self.keyframes[-1]
        kf.gdesc = np.asarray(self.place.describe(rgb), np.float32)
        if self._loop_cooldown > 0:
            self._loop_cooldown -= 1
            return None
        cands = [k for k in self.keyframes[:-1] if k.gdesc is not None and kf.id - k.id >= cfg.loop_min_gap_kf
                 and timestamp - k.timestamp >= cfg.loop_min_gap_s]
        if not cands:
            return None
        sims = np.array([float(kf.gdesc @ k.gdesc) for k in cands])
        order = [i for i in np.argsort(-sims) if sims[i] >= cfg.loop_min_similarity]
        if not order:
            return None
        sigma = self.relative_sigma(kf.id)
        chosen = order[:cfg.loop_top_k]
        chosen += [i for i in order[cfg.loop_top_k:] if sigma.get(cands[i].id, 0.0) >= cfg.loop_min_sigma][
            :cfg.loop_uncertain_k]
        rank = {i: r for r, i in enumerate(order)}
        for idx in chosen:
            entry = {"t": float(timestamp), "kf": kf.id, "candidate": cands[idx].id, "rank": rank[idx],
                     "t_candidate": float(cands[idx].timestamp), "similarity": float(sims[idx]),
                     "sigma_m": float(sigma.get(cands[idx].id, float("nan")))}
            result = self._verify_loop(kf, cands[idx], entry)
            self.loop_log.append(entry)
            del self.loop_log[:-2000]
            if result is not None:
                self._loop_cooldown = cfg.loop_cooldown_kf
                return result
        return None

    def _note_covisibility(self, kf, matched_lm):
        """Edges to the older keyframes whose landmarks this keyframe was tracked on."""
        cfg = self.cfg
        lm = matched_lm[matched_lm >= 0]
        if len(lm) == 0:
            return
        anchors = self.lm.anchor[lm]
        anchors = anchors[(anchors >= 0) & (anchors < kf.id - 1)]
        if len(anchors) == 0:
            return
        ids, counts = np.unique(anchors, return_counts=True)
        by = {k.id: k for k in self.keyframes}
        for i in np.argsort(-counts)[:cfg.covis_max_per_kf]:
            a, n = int(ids[i]), int(counts[i])
            if n >= cfg.covis_min_shared and a in by:
                self.covis_edges.append((a, kf.id, between(by[a].pose, kf.pose), n))

    def relative_sigma(self, source_id):
        """Position uncertainty (m, 1-sigma) of every keyframe relative to keyframe
        ``source_id`` along the cheapest chain of pose-graph edges: odometry between
        consecutive keyframes ((0.01 + 0.03 d)^2 per hop, as in ``_close_loop``) and
        accepted loop edges (0.03^2). Regions already tied in by a loop are close."""
        import heapq

        ordered = sorted(self.keyframes, key=lambda k: k.id)
        adj = {k.id: [] for k in ordered}
        for a, b in zip(ordered[:-1], ordered[1:]):
            d = float(np.hypot(*(b.pose[:2] - a.pose[:2])))
            var = (0.01 + 0.03 * d) ** 2
            adj[a.id].append((b.id, var))
            adj[b.id].append((a.id, var))
        for i, j, _ in self.loop_edges + self.shadow_edges:
            if i in adj and j in adj:
                adj[i].append((j, 0.03 ** 2))
                adj[j].append((i, 0.03 ** 2))
        for i, j, _, _ in self.covis_edges:
            if i in adj and j in adj:
                adj[i].append((j, 0.02 ** 2))
                adj[j].append((i, 0.02 ** 2))
        best = {source_id: 0.0}
        heap = [(0.0, source_id)]
        while heap:
            v, u = heapq.heappop(heap)
            if v > best.get(u, np.inf):
                continue
            for w, var in adj.get(u, ()):
                nv = v + var
                if nv < best.get(w, np.inf):
                    best[w] = nv
                    heapq.heappush(heap, (nv, w))
        return {i: float(np.sqrt(v)) for i, v in best.items()}

    def _path_length(self, i, j):
        ids = sorted(k.id for k in self.keyframes if i <= k.id <= j)
        by = {k.id: k for k in self.keyframes}
        return float(sum(np.hypot(*(by[b].pose[:2] - by[a].pose[:2])) for a, b in zip(ids[:-1], ids[1:])))

    def _verify_loop(self, kf, cand, entry):
        cfg = self.cfg
        near = self.lm.alive & (np.abs(self.lm.anchor - cand.id) <= cfg.loop_window_kf) & (self.lm.anchor >= 0)
        wide = self.lm.alive & (np.abs(self.lm.anchor - cand.id) <= 2 * cfg.loop_window_kf) & (self.lm.anchor >= 0)
        found = self._global_matches(kf.pts, kf.desc, candidates=np.flatnonzero(near))
        if found is None:
            entry["outcome"] = "too_few_landmarks"
            return None
        kp, lm, uniq = found
        hyps = self._planar_hypotheses(kf.pts, kp, lm, uniq)
        if len(hyps) == 0:
            entry["outcome"] = "no_hypothesis"
            return None
        P, uv_q = self.lm.pos[lm], kf.pts[kp]
        scores, _ = self._score_poses(hyps, P, uv_q, kp)
        modes = []
        for h in np.argsort(-scores):
            if scores[h] < 8:
                break
            if any(np.hypot(*(hyps[h, :2] - m[:2])) < 0.15 and abs(wrap(hyps[h, 2] - m[2])) < math.radians(10)
                   for m in modes):
                continue
            modes.append(hyps[h])
            if len(modes) >= cfg.reloc_modes:
                break
        checked = []
        for m in modes:
            _, okm = self._score_poses(m[None], P, uv_q, kp)
            if okm[0].sum() < 6:
                continue
            refined, _, _ = self.optimize(m, P[okm[0]], uv_q[okm[0]])
            v = self._verify(refined, kf.pts, kf.desc, candidates=np.flatnonzero(wide))
            if v is not None:
                checked.append(v)
        if not checked:
            entry["outcome"] = "not_verified"
            return None
        checked.sort(key=lambda c: -c["inliers"])
        best = checked[0]
        rivals = [c for c in checked[1:] if np.hypot(*(c["pose"][:2] - best["pose"][:2])) > 0.15
                  or abs(wrap(c["pose"][2] - best["pose"][2])) > math.radians(10)]
        second = rivals[0]["inliers"] if rivals else 0
        jump = float(np.hypot(*(best["pose"][:2] - kf.pose[:2])))
        travel = self._path_length(cand.id, kf.id)
        if travel < cfg.loop_min_travel:
            entry["outcome"] = "rejected_not_a_revisit"
            entry["travel_m"] = travel
            return None
        allowed = cfg.loop_max_correction + cfg.loop_correction_per_m * travel
        entry.update({"verified": best["inliers"], "explained": round(best["explained"], 3), "second": second,
                      "pose_loop": [float(v) for v in best["pose"]], "pose_before": [float(v) for v in kf.pose],
                      "pose_candidate": [float(v) for v in cand.pose],
                      "correction_m": jump, "travel_m": travel})
        if (best["inliers"] < cfg.loop_verify_min or best["explained"] < cfg.reloc_explained_min
                or best["inliers"] < cfg.reloc_distinct_ratio * second):
            entry["outcome"] = "rejected_verification"
            return None
        if jump > allowed:
            entry["outcome"] = "rejected_implausible_correction"
            return None
        partner = self._confirm_loop(kf.id, cand.id, between(kf.pose, best["pose"]))
        if partner is None:
            entry["outcome"] = "awaiting_confirmation"
            return None
        entry["confirmed_by_kf"] = partner["kf"]
        z = between(cand.pose, best["pose"])
        return self._close_loop(kf, cand, z, entry)

    def _confirm_loop(self, kf_id, cand_id, corr):
        """Temporal consistency: return an earlier pending hypothesis (within
        loop_confirm_window_kf keyframes, same old region) implying the same
        correction ``corr`` of the current pose, or store this one and return None."""
        cfg = self.cfg
        self._loop_pending = [p for p in self._loop_pending if 0 <= kf_id - p["kf"] <= cfg.loop_confirm_window_kf]
        partner = next((p for p in self._loop_pending if 1 <= kf_id - p["kf"]
                        and abs(p["candidate"] - cand_id) <= 2 * cfg.loop_window_kf
                        and np.hypot(*(p["corr"][:2] - corr[:2])) <= cfg.loop_confirm_m
                        and abs(wrap(p["corr"][2] - corr[2])) <= cfg.loop_confirm_rad), None)
        if partner is None:
            self._loop_pending.append({"kf": kf_id, "candidate": cand_id, "corr": np.asarray(corr, float)})
            return None
        self._loop_pending.clear()
        return partner

    def _close_loop(self, kf, cand, z, entry):
        cfg = self.cfg
        g = PoseGraph()
        ordered = sorted(self.keyframes, key=lambda k: k.id)
        for k in ordered:
            g.add_node(k.id, k.pose, fixed=(k.id == ordered[0].id or k.id < self.n_loaded_keyframes))
        for a, b in zip(ordered[:-1], ordered[1:]):
            rel = between(a.pose, b.pose)
            d = float(np.hypot(rel[0], rel[1]))
            sx = 0.01 + 0.03 * d
            g.add_edge(a.id, b.id, rel, (sx, sx, 0.01 + 0.02 * abs(rel[2])))
        for i, j, zz in self.loop_edges + self.shadow_edges:
            g.add_edge(i, j, zz, (0.03, 0.03, 0.02), loop=True)
        for i, j, zz, _ in self.covis_edges:
            if i in g.poses and j in g.poses:
                g.add_edge(i, j, zz, (0.02, 0.02, 0.01))
        g.add_edge(cand.id, kf.id, z, (0.03, 0.03, 0.02), loop=True)
        before = {k.id: k.pose.copy() for k in ordered}
        after = g.optimize()
        distortion = trajectory_distortion(before, after)
        weight = g.edges[-1].weight
        entry.update({"distortion_m": distortion, "loop_weight": weight})
        if distortion > cfg.loop_distortion_max or weight < 0.5:
            entry["outcome"] = "rejected_pose_graph"
            return None
        if cfg.loop_shadow:
            self.shadow_edges.append((cand.id, kf.id, z))
            entry.update({"outcome": "shadow_closed", "z": [float(v) for v in z]})
            return entry
        corrections = {i: correction(before[i], after[i]) for i in before}
        by = {k.id: k for k in self.keyframes}
        for i in corrections:
            by[i].pose = after[i].copy()
        # landmarks move with the keyframe that created them
        for i, T in corrections.items():
            sel = self.lm.anchor == i
            if sel.any() and (abs(T[2]) > 1e-9 or abs(T[0]) > 1e-9 or abs(T[1]) > 1e-9):
                self.lm.pos[sel, :2] = apply_transform(T, self.lm.pos[sel, :2])
        T_cur = corrections[kf.id]
        self.pose = apply_to_pose(T_cur, self.pose)
        c, s_ = math.cos(T_cur[2]), math.sin(T_cur[2])
        self.velocity[:2] = [c * self.velocity[0] - s_ * self.velocity[1], s_ * self.velocity[0] + c * self.velocity[1]]
        self._motion_log.clear()  # the jump is a correction, not motion
        self._dr = None
        self._kp_tree = self._kp_tree_pts = None
        self.loop_edges.append((cand.id, kf.id, z))
        entry.update({"outcome": "closed", "pose_after": [float(v) for v in self.pose]})
        if self.on_loop_closure is not None:
            self.on_loop_closure(corrections, {k.id: k.timestamp for k in self.keyframes})
        return entry

    # ------------------------------------------------------------ planar relocalisation
    def _global_matches(self, pts, desc, candidates=None):
        """Whole-map descriptor matches with a ratio test against the best match at a
        DIFFERENT place (near-duplicate landmarks of one point are not ambiguity).
        Returns (kp, lm, unique): a keypoint ambiguous between two places yields both
        pairs with unique=False (scored, but never used to seed a hypothesis).
        ``candidates``: restrict to these landmark indices (loop closure)."""
        alive = np.flatnonzero(self.lm.alive) if candidates is None else np.asarray(candidates, int)
        if len(alive) < 50 or len(pts) < 30:
            return None
        k = self.cfg.reloc_knn
        knn = self.matcher.knnMatch(desc, self.lm.desc[alive], k=k)
        rows = [m for m in knn if len(m) == k and m[0].distance <= self.cfg.reloc_max_hamming]
        if not rows:
            return None
        q = np.array([m[0].queryIdx for m in rows])
        idx = alive[np.array([[c.trainIdx for c in m] for m in rows])]  # (Q, k)
        dist = np.array([[c.distance for c in m] for m in rows], float)
        P = self.lm.pos[idx]  # (Q, k, 3)
        sep = self.cfg.reloc_distinct_m
        away = np.linalg.norm(P - P[:, :1], axis=2) > sep  # (Q, k): a different place than the best
        has_other = away.any(1)
        first_other = np.argmax(away, 1)
        other_d = np.where(has_other, dist[np.arange(len(q)), first_other], np.inf)
        unique = dist[:, 0] < self.cfg.reloc_ratio * other_d
        # Ambiguous between exactly two places: every comparably good candidate is at
        # the best place or within sep of the first other place.
        P2 = P[np.arange(len(q)), first_other]
        close = dist < dist[:, :1] / self.cfg.reloc_ratio
        at_second = np.linalg.norm(P - P2[:, None], axis=2) <= sep
        two = ~unique & has_other & np.all(~close | ~away | at_second, axis=1)
        kp = np.concatenate([q[unique], q[two], q[two]])
        lm = np.concatenate([idx[unique, 0], idx[two, 0], idx[two][np.arange(two.sum()), first_other[two]]])
        uniq = np.concatenate([np.ones(unique.sum(), bool), np.zeros(2 * two.sum(), bool)])
        if len(kp) == 0:
            return None
        return kp, lm, uniq

    def _project_batch(self, P, poses):
        """Project world points P (M,3) for planar poses (H,3) -> uv (H,M,2), depth (H,M)."""
        x, y, th = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
        c, s_ = np.cos(th), np.sin(th)
        dx, dy, dz = P[None, :, 0] - x, P[None, :, 1] - y, np.broadcast_to(P[None, :, 2] - self.model.base_z,
                                                                           (len(poses), len(P)))
        pb = np.stack([c * dx + s_ * dy, -s_ * dx + c * dy, dz], -1)
        R_cb, t_cb = self._T_cb[:3, :3], self._T_cb[:3, 3]
        pc = pb @ R_cb.T + t_cb
        z = pc[..., 2]
        zs = np.where(z > 0.02, z, 0.02)
        K = self.model.K
        uv = np.stack([K[0, 0] * pc[..., 0] / zs + K[0, 2], K[1, 1] * pc[..., 1] / zs + K[1, 2]], -1)
        return uv, z

    def _score_poses(self, poses, P, uv_q, kp):
        """Unique keypoints whose matched landmark reprojects within score_px, per pose."""
        proj, z = self._project_batch(P, poses)
        ok = (np.linalg.norm(proj - uv_q[None], axis=2) < self.cfg.reloc_score_px) & (z > 0.05)
        scores = np.zeros(len(poses), int)
        for h in range(len(poses)):
            scores[h] = len(np.unique(kp[ok[h]]))
        return scores, ok

    def _planar_hypotheses(self, pts, kp, lm, uniq, pool=False):
        """RANSAC over pairs of floor correspondences (closed-form planar rigid fit).
        ``pool``: also use floor seeds of the last frames of a relocalisation turn,
        moved into the current robot frame by the commanded motion since."""
        q, ok = self._ipm(pts[kp], np.zeros(3))
        rng_q = np.hypot(q[:, 0], q[:, 1])
        sel = uniq & ok & (self.lm.kind[lm] == 0) & (rng_q <= self.cfg.reloc_hyp_range)
        Q, L, R, tag = q[sel, :2], lm[sel], rng_q[sel], [(0, int(k)) for k in kp[sel]]
        if pool and self.cfg.reloc_pool_frames > 0:
            odom = self._reloc_odom.copy()
            for f, (o, qq, ll, rr) in enumerate(self._reloc_pool, start=1):
                rel = between(odom, o)  # the earlier robot frame expressed in the current one
                c0, s0 = math.cos(rel[2]), math.sin(rel[2])
                moved = np.column_stack([c0 * qq[:, 0] - s0 * qq[:, 1] + rel[0], s0 * qq[:, 0] + c0 * qq[:, 1] + rel[1]])
                Q, L, R = np.vstack([Q, moved]), np.r_[L, ll], np.r_[R, rr]
                tag += [(f, k) for k in range(len(qq))]
            self._reloc_pool.append((odom, q[sel, :2].copy(), lm[sel].copy(), rng_q[sel].copy()))
            del self._reloc_pool[:-self.cfg.reloc_pool_frames]
        if len(Q) < 2:
            return np.zeros((0, 3))
        i, j = np.triu_indices(len(Q), 1)
        lq = np.hypot(*(Q[i] - Q[j]).T)
        lmap = np.hypot(*(self.lm.pos[L[i], :2] - self.lm.pos[L[j], :2]).T)
        tol = 0.02 + 0.04 * np.maximum(R[i], R[j])
        same = np.array([tag[x] == tag[y] for x, y in zip(i, j)], bool) if len(i) else np.zeros(0, bool)
        keep = (lq >= 0.12) & (np.abs(lq - lmap) <= tol) & ~same & (L[i] != L[j])
        a, b = i[keep], j[keep]
        if len(a) == 0:
            return np.zeros((0, 3))
        if len(a) > self.cfg.reloc_iters:
            pick = np.random.default_rng(self.frames).choice(len(a), self.cfg.reloc_iters, replace=False)
            a, b = a[pick], b[pick]
        q, lm = np.column_stack([Q, np.zeros(len(Q))]), L
        vq = q[b, :2] - q[a, :2]
        vm = self.lm.pos[lm[b], :2] - self.lm.pos[lm[a], :2]
        th = np.arctan2(vm[:, 1], vm[:, 0]) - np.arctan2(vq[:, 1], vq[:, 0])
        c, s_ = np.cos(th), np.sin(th)
        mq = 0.5 * (q[a, :2] + q[b, :2])
        mm = 0.5 * (self.lm.pos[lm[a], :2] + self.lm.pos[lm[b], :2])
        tx = mm[:, 0] - (c * mq[:, 0] - s_ * mq[:, 1])
        ty = mm[:, 1] - (s_ * mq[:, 0] + c * mq[:, 1])
        return np.stack([tx, ty, np.arctan2(s_, c)], 1)

    def _verify(self, pose, pts, desc, candidates=None):
        """Guided re-match at a candidate pose (as tracking does) and how much of what
        the map predicts to be visible was found again (image cells, so duplicate
        landmarks of one point count once)."""
        alive = np.flatnonzero(self.lm.alive) if candidates is None else np.asarray(candidates, int)
        d = self.lm.pos[alive, :2] - pose[:2]
        rel = np.arctan2(d[:, 1], d[:, 0]) - pose[2]
        rel = np.arctan2(np.sin(rel), np.cos(rel))
        self._frame_candidates = alive[(np.hypot(d[:, 0], d[:, 1]) < 4.0) & (np.abs(rel) < math.radians(70))]
        saved_prior, self._motion_prior = self._motion_prior, None
        try:
            lm2, kp2 = self._guided(pose, pts, desc, 8.0, count_visible=False)
            if len(lm2) < self.cfg.min_inliers:
                return None
            pose2, inl, H, err = self._pose_from_matches(pose, lm2, kp2, pts)
            if H is None:
                return None
            # predicted-visible confirmed landmarks within 2.5 m, in 16 px image cells
            pred = self._frame_candidates[self.lm.confirmed[self._frame_candidates]]
            proj, z = self._project(self.lm.pos[pred], pose2)
            dist = np.hypot(*(self.lm.pos[pred, :2] - pose2[:2]).T)
            W, Hh = self.model.width, self.model.height
            vis = (z > 0.05) & (dist < 2.5) & (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < Hh)
            cells = {(int(u) // 16, int(v_) // 16) for u, v_ in proj[vis]}
            found = {(int(u) // 16, int(v_) // 16) for u, v_ in pts[kp2[inl]]}
            explained = len(cells & found) / max(1, len(cells))
            return {"pose": pose2, "inliers": int(inl.sum()), "H": H, "explained": float(explained),
                    "triangulated": int((self.lm.kind[lm2[inl]] == 1).sum()),
                    "mean_px": float(np.mean(err[inl])) if inl.any() else None}
        finally:
            self._motion_prior = saved_prior
            self._frame_candidates = None

    def _global_localize_planar(self, pts, desc, candidates=None, pool=False):
        found = self._global_matches(pts, desc, candidates=candidates)
        if found is None:
            return None
        kp, lm, uniq = found
        hyps = self._planar_hypotheses(pts, kp, lm, uniq, pool=pool)
        info = {"matches": int(len(kp)), "unique": int(uniq.sum()), "hypotheses": int(len(hyps))}
        self._reloc_info = info
        if len(hyps) == 0:
            return None
        P, uv_q = self.lm.pos[lm], pts[kp]
        scores, ok = self._score_poses(hyps, P, uv_q, kp)
        order = np.argsort(-scores)
        modes = []
        for h in order:
            if scores[h] < 8:
                break
            if any(np.hypot(*(hyps[h, :2] - m[:2])) < 0.15 and abs(wrap(hyps[h, 2] - m[2])) < math.radians(10)
                   for m in modes):
                continue
            modes.append(hyps[h])
            if len(modes) >= self.cfg.reloc_modes:
                break
        checked = []
        for m in modes:
            _, okm = self._score_poses(m[None], P, uv_q, kp)
            sel = okm[0]
            if sel.sum() < 6:
                continue
            refined, err, _ = self.optimize(m, P[sel], uv_q[sel])
            v = self._verify(refined, pts, desc)
            if v is not None:
                checked.append(v)
        if not checked:
            return None
        checked.sort(key=lambda c: -c["inliers"])
        best = checked[0]
        # distinct places only: a second verified mode at the same place is not a rival
        rivals = [c for c in checked[1:] if np.hypot(*(c["pose"][:2] - best["pose"][:2])) > 0.15
                  or abs(wrap(c["pose"][2] - best["pose"][2])) > math.radians(10)]
        second = rivals[0]["inliers"] if rivals else 0
        info.update({"verified": best["inliers"], "explained": round(best["explained"], 3),
                     "second": second, "triangulated": best["triangulated"]})
        if (best["inliers"] < self.cfg.reloc_verify_min or best["explained"] < self.cfg.reloc_explained_min
                or best["inliers"] < self.cfg.reloc_distinct_ratio * second):
            info["rejected"] = True
            return None
        return best["pose"], best["inliers"], best["H"]

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
                            confirmed=self.lm.confirmed[alive], anchor=self.lm.anchor[alive],
                            flags=(self.lm.flags[alive] & 2).astype(np.int8))
        described = [kf for kf in self.keyframes if kf.gdesc is not None]
        if described:  # v2: place descriptors for keyframe retrieval (relocalisation, loop closure)
            np.savez_compressed(directory / "keyframes.npz", ids=np.array([kf.id for kf in described]),
                                gdesc=np.stack([kf.gdesc for kf in described]).astype(np.float16))
        meta = {
            "schema": "amr_rl.vslam-map.v2",
            "place_descriptor": getattr(self.place, "name", None) if described else None,
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
        if "anchor" in data:  # v2
            slam.lm.anchor[:] = data["anchor"]
        if "flags" in data:  # validated floor points (C2a)
            slam.lm.flags[:] = data["flags"]
        slam.status = RELOCALIZING
        slam.n_loaded_keyframes = len(meta["keyframes"])
        for item in meta["keyframes"]:
            slam.keyframes.append(Keyframe(item["id"], item["t"], np.array(item["pose"]),
                                           np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0, int)))
        if (directory / "keyframes.npz").exists():
            kd = np.load(directory / "keyframes.npz")
            by = {kf.id: kf for kf in slam.keyframes}
            for i, g in zip(kd["ids"], kd["gdesc"]):
                if int(i) in by:
                    by[int(i)].gdesc = g.astype(np.float32)
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
