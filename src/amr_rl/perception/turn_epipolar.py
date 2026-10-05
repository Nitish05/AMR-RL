"""Depth-free in-place-turn estimator from two-view epipolar geometry.

ENGINEERED geometric estimator (nothing here is learned). During an in-place turn
the base rotates by ``theta`` about the axle centre (base_link z axis) and the
camera, mounted ``camera_offset`` ahead of the axle, moves on a known circle. The
relative camera pose between a reference frame A and a frame B is therefore a
function of ``theta`` alone, through the fixed camera->base calibration
(``CameraModel.T_base_cam``). The two-view epipolar constraint

    f_i(theta) = b_B,i^T  E(theta)  b_A,i = 0,      E = [t]_x R

then has ONE unknown, so every feature correspondence yields theta (up to a few
discrete roots) without any landmark depth. This avoids the near-wall
over-rotation of the map-based tracker, whose floor-lifted wall points carry wrong
depth.

Method: 1-point RANSAC over per-correspondence roots (Scaramuzza, "1-Point-RANSAC
Structure from Motion for Vehicle-Mounted Cameras by Exploiting Non-holonomic
Constraints", IJCV 95(1), 2011), scored by Sampson error; the inlier set is then
refined by a bounded 1-D minimisation of a Cauchy-robust Sampson cost (planar
minimal-solver idea as in Choi & Kim, "Fast and reliable minimal relative pose
estimation under planar motion", Image and Vision Computing 69, 2018).

Exact reduction used here: with phi = theta / 2 and the camera offset ``c`` in
base_link, the camera translation is (Rz(-theta) - I) c = 2 sin(phi) M(phi) c with
M linear in (cos phi, sin phi). Dropping the common factor 2 sin(phi) (which would
make every residual vanish at theta = 0) gives a smooth reduced essential matrix
that is an odd trigonometric polynomial in phi on the basis
{cos phi, sin phi, cos 3phi, sin 3phi}. Residuals and Sampson errors are therefore
evaluated for many correspondences and angles at once by a 4-term expansion.

Conventions follow ``camera_model``: theta is the planar yaw change of base_link
from frame A to frame B (positive = counter-clockwise, left turn), pixel inputs
are OpenCV pixels. Only the RGB camera calibration and image matches are used;
no simulator state (privilege boundary).

Accuracy is limited by geometry, not by the solver: with a lateral camera
translation the epipolar lines are near-horizontal, so the horizontal image motion
of a yaw is largely absorbed by unknown depth along those lines; theta is
constrained mainly through off-axis, off-horizon points (observable because the
camera is pitched down). Measured on synthetic data with this camera (320x240,
fx = 208 px), 300 matches, 30 % outliers, 0.5 px noise: RMS error 0.08-0.2 deg
(worst of 30 seeds 0.15-0.45 deg), unbiased with respect to scene depth (near
wall at 0.18-0.4 m or far scene alike); 0.1 px noise gives < 0.07 deg. The
returned ``sigma`` tracks the measured spread.

Base slides: a pure-turn fit is badly biased by a forward/backward slide in a
near scene (up to ~9 deg for 2 cm at a 0.2-0.4 m wall). Each estimate is therefore
compared with a general planar model (theta + free translation direction, robust
2-parameter fit); when that explains clearly more matches the planar theta is
returned with ``translation_detected=True``. With the check, 2 cm slides in any
direction cost <= ~1 deg; lateral slides are not detectable but bias the turn
model little (< ~0.7 deg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def turn_relative_pose(model, theta: float, base_translation=(0.0, 0.0)) -> np.ndarray:
    """4x4 ``T_B_A``: maps points from camera-A optical coordinates to camera-B
    optical coordinates when the base moves from planar pose (0, 0, 0) to
    (dx, dy, theta) (dx, dy expressed in base frame A). ``theta`` only is the
    in-place-turn model; the translation exists for analysis and tests."""
    T_base_cam = np.asarray(model.T_base_cam, float)
    T_cam_base = np.asarray(model.T_cam_base, float)
    P = np.eye(4)
    P[:3, :3] = _rz(theta)
    P[0, 3], P[1, 3] = float(base_translation[0]), float(base_translation[1])
    # X_world = P_A T_base_cam X_A = P_B T_base_cam X_B with P_A = I.
    return T_cam_base @ np.linalg.inv(P) @ T_base_cam


def essential_from_turn(model, theta: float, base_translation=(0.0, 0.0)) -> np.ndarray:
    """Essential matrix E = [t]_x R of ``turn_relative_pose`` (b_B^T E b_A = 0), with
    t scaled to unit norm (zero matrix when there is no camera translation)."""
    T = turn_relative_pose(model, theta, base_translation)
    t = T[:3, 3]
    n = np.linalg.norm(t)
    return _skew(t / n) @ T[:3, :3] if n > 1e-12 else np.zeros((3, 3))


def _reduced_essential(R_bc: np.ndarray, c: np.ndarray, phi: float) -> np.ndarray:
    """E(theta) / (2 sin phi), theta = 2 phi (smooth through theta = 0)."""
    cp, sp = math.cos(phi), math.sin(phi)
    M = np.array([[-sp, cp, 0.0], [-cp, -sp, 0.0], [0.0, 0.0, 0.0]])
    R = R_bc.T @ _rz(-2.0 * phi) @ R_bc
    t = R_bc.T @ (M @ c)
    return _skew(t) @ R


def _basis(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, float)
    return np.stack([np.cos(phi), np.sin(phi), np.cos(3 * phi), np.sin(3 * phi)])


@dataclass
class TurnEstimate:
    """Result of :meth:`TurnEpipolar.estimate` (engineered, not learned)."""

    theta: float  # base yaw change A -> B, rad (CCW positive); nan when nothing was found
    sigma: float  # 1-sigma standard error from cost curvature x residual variance, rad
    n_inliers: int
    inliers: np.ndarray = field(repr=False)  # bool mask over the input matches
    ok: bool
    reason: str = ""
    n_matches: int = 0
    rms_px: float = float("nan")  # inlier Sampson RMS in pixels (fx units)
    # Planar consistency check (general planar motion: theta + free translation direction).
    # When it explains clearly more matches than the pure turn, the base slid; ``theta``
    # is then the planar-model value and ``model`` is "planar".
    translation_detected: bool = False
    model: str = "turn"
    theta_turn: float = float("nan")  # pure-turn (1-DOF) value, always reported


@dataclass
class TurnEpipolarConfig:
    threshold_px: float = 1.5  # Sampson inlier threshold, pixels
    max_abs: float = 1.0  # search window without a hint, rad
    hint_window_min: float = 0.2  # with a hint: +-max(hint_window_min, hint_window_frac*|hint|)
    hint_window_frac: float = 0.5
    min_matches: int = 10
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25
    max_sigma: float = 0.01  # rad; larger => degenerate / flat cost
    grid_step: float = 0.004  # rad of theta for the per-correspondence root search
    max_hypotheses: int = 400
    cauchy_scale_px: float = 1.0
    seed: int = 0
    check_translation: bool = True  # fit the 2-DOF planar model and compare inlier counts
    translation_gain_abs: int = 8  # planar inliers must exceed turn inliers by max(abs,
    translation_gain_frac: float = 0.05  # frac * turn inliers) to declare a slide


class TurnEpipolar:
    """1-point-RANSAC in-place-turn estimator for one calibrated camera.

    ``estimate(uv_a, uv_b, theta_hint)`` takes matched pixels (N x 2 each) from a
    reference frame A and a current frame B and returns a :class:`TurnEstimate`.
    """

    def __init__(self, model, config: TurnEpipolarConfig | None = None):
        self.model = model
        self.cfg = config or TurnEpipolarConfig()
        self.K = np.asarray(model.K, float)
        self.Kinv = np.linalg.inv(self.K)
        self.fx = float(self.K[0, 0])
        T = np.asarray(model.T_base_cam, float)
        R_bc, c = T[:3, :3], T[:3, 3].copy()
        self._R_cb = R_bc.T
        c[2] = 0.0  # vertical offset is unchanged by a yaw (and cancels in the translation)
        if np.linalg.norm(c[:2]) < 1e-6:
            raise ValueError("camera on the turning axis: in-place turns carry no translation")
        # Exact 4-term expansion E_red(phi) = sum_k basis_k(phi) * E_k (least squares over samples).
        phis = np.linspace(-math.pi, math.pi, 13, endpoint=False) + 0.1
        A = _basis(phis).T  # (13, 4)
        Es = np.stack([_reduced_essential(R_bc, c, p).ravel() for p in phis])  # (13, 9)
        coef, *_ = np.linalg.lstsq(A, Es, rcond=None)
        if np.abs(A @ coef - Es).max() > 1e-9:  # pragma: no cover - guards the derivation
            raise RuntimeError("reduced essential expansion failed")
        self._Ek = coef.reshape(4, 3, 3)

    # ------------------------------------------------------------------ helpers
    def bearings(self, uv: np.ndarray) -> np.ndarray:
        """Normalised homogeneous image coordinates (z = 1) of pixels."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.c_[uv, np.ones(len(uv))] @ self.Kinv.T

    def essential(self, theta: float) -> np.ndarray:
        """Reduced essential matrix (scale differs from ``essential_from_turn``)."""
        return np.tensordot(_basis(theta / 2.0), self._Ek, axes=1)

    def _terms(self, xa: np.ndarray, xb: np.ndarray):
        Ea = np.einsum("kij,nj->kni", self._Ek, xa)  # (4, N, 3)  E_k x_a
        Etb = np.einsum("kij,ni->knj", self._Ek, xb)  # (4, N, 3)  E_k^T x_b
        g = np.einsum("kni,ni->kn", Ea, xb)  # (4, N)
        return g, Ea[:, :, :2], Etb[:, :, :2]

    @staticmethod
    def _sampson(terms, theta: np.ndarray) -> np.ndarray:
        """Signed Sampson residuals (normalised units), shape (len(theta), N)."""
        g, Ea, Etb = terms
        B = _basis(np.atleast_1d(theta) / 2.0)  # (4, H)
        r = B.T @ g  # (H, N)
        la = np.einsum("kh,kni->hni", B, Ea)
        lb = np.einsum("kh,kni->hni", B, Etb)
        den = (la ** 2).sum(-1) + (lb ** 2).sum(-1)
        return r / np.sqrt(np.maximum(den, 1e-18))

    def window(self, theta_hint: float | None) -> tuple[float, float]:
        cfg = self.cfg
        if theta_hint is None or not np.isfinite(theta_hint):
            return -cfg.max_abs, cfg.max_abs
        w = max(cfg.hint_window_min, cfg.hint_window_frac * abs(float(theta_hint)))
        return float(theta_hint) - w, float(theta_hint) + w

    def _roots(self, g: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """All roots of f_i(theta) = 0 in [lo, hi] for every correspondence (flattened)."""
        n = max(int(math.ceil((hi - lo) / self.cfg.grid_step)), 2)
        grid = np.linspace(lo, hi, n + 1)
        F = _basis(grid / 2.0).T @ g  # (G, N)
        s0, s1 = F[:-1], F[1:]
        gi, ni = np.nonzero((s0 == 0) | (np.sign(s0) * np.sign(s1) < 0))
        if gi.size == 0:
            return np.empty(0)
        a, b = grid[gi], grid[gi + 1]
        fa, fb = s0[gi, ni], s1[gi, ni]
        for _ in range(3):  # vectorised regula falsi (Illinois-free: interval is tiny)
            denom = np.where(fb - fa == 0, 1e-300, fb - fa)
            m = a - fa * (b - a) / denom
            fm = np.einsum("kh,kh->h", _basis(m / 2.0), g[:, ni])
            left = np.sign(fm) == np.sign(fa)
            a, fa = np.where(left, m, a), np.where(left, fm, fa)
            b, fb = np.where(left, b, m), np.where(left, fb, fm)
        denom = np.where(fb - fa == 0, 1e-300, fb - fa)
        return a - fa * (b - a) / denom

    # ----------------------------------------------------------------- estimate
    def estimate(self, uv_a, uv_b, theta_hint: float | None = None) -> TurnEstimate:
        cfg = self.cfg
        uv_a = np.asarray(uv_a, float).reshape(-1, 2)
        uv_b = np.asarray(uv_b, float).reshape(-1, 2)
        n = len(uv_a)
        empty = np.zeros(n, bool)

        def fail(reason, theta=float("nan"), sigma=float("inf"), inl=empty, rms=float("nan")):
            return TurnEstimate(theta, sigma, int(inl.sum()), inl, False, reason, n, rms, theta_turn=theta)

        if len(uv_b) != n:
            raise ValueError("uv_a and uv_b must have the same length")
        if n < cfg.min_matches:
            return fail("too few matches")
        terms = self._terms(self.bearings(uv_a), self.bearings(uv_b))
        lo, hi = self.window(theta_hint)
        hyps = self._roots(terms[0], lo, hi)
        if hyps.size == 0:
            return fail("no root in window")
        if hyps.size > cfg.max_hypotheses:
            hyps = np.random.default_rng(cfg.seed).choice(hyps, cfg.max_hypotheses, replace=False)
        thr = cfg.threshold_px / self.fx
        err2 = self._sampson(terms, hyps) ** 2  # (H, N)
        score = np.minimum(err2, thr ** 2).sum(1)  # MSAC
        best = float(hyps[int(np.argmin(score))])

        # Robust refinement around the best hypothesis (Cauchy on Sampson residuals).
        cs2 = (cfg.cauchy_scale_px / self.fx) ** 2
        inl = err2[int(np.argmin(score))] < thr ** 2
        theta = best
        for _ in range(2):
            idx = np.flatnonzero(inl)
            if idx.size < 2:
                break
            sub = tuple(x[:, idx] for x in terms)

            def cost(th, sub=sub):
                r2 = self._sampson(sub, np.array([th]))[0] ** 2
                return float(np.log1p(r2 / cs2).sum())

            span = max(4 * cfg.grid_step, 0.02)
            res = minimize_scalar(cost, bounds=(max(lo, theta - span), min(hi, theta + span)),
                                  method="bounded", options={"xatol": 1e-7})
            theta = float(res.x)
            inl = self._sampson(terms, np.array([theta]))[0] ** 2 < thr ** 2

        k = int(inl.sum())
        r = self._sampson(terms, np.array([theta]))[0][inl]
        h = 1e-5
        dr = (self._sampson(terms, np.array([theta + h]))[0][inl]
              - self._sampson(terms, np.array([theta - h]))[0][inl]) / (2 * h)
        info = float((dr ** 2).sum())
        var_r = float((r ** 2).sum() / max(k - 1, 1))
        sigma = math.sqrt(var_r / info) if info > 0 else float("inf")
        rms = float(math.sqrt((r ** 2).mean()) * self.fx) if k else float("nan")
        if k < cfg.min_inliers:
            result = fail("too few inliers", theta, sigma, inl, rms)
        elif k < cfg.min_inlier_ratio * n:
            result = fail("low inlier ratio", theta, sigma, inl, rms)
        elif not np.isfinite(sigma) or sigma > cfg.max_sigma:
            result = fail("flat cost (degenerate)", theta, sigma, inl, rms)
        elif min(theta - lo, hi - theta) < 1e-4:
            result = fail("estimate at window edge", theta, sigma, inl, rms)
        else:
            result = TurnEstimate(theta, sigma, k, inl, True, "", n, rms, theta_turn=theta)
        if cfg.check_translation and k >= 2:
            self._planar_check(result, self.bearings(uv_a), self.bearings(uv_b), thr, lo, hi)
        return result

    # ------------------------------------------------------- planar (2-DOF) check
    def _planar_residuals(self, p, xa, xb):
        th, al = p
        R = self._R_cb @ _rz(-th) @ self._R_cb.T
        E = _skew(self._R_cb @ np.array([math.cos(al), math.sin(al), 0.0])) @ R
        la, lb = xa @ E.T, xb @ E
        r = np.einsum("ni,ni->n", la, xb)
        den = la[:, 0] ** 2 + la[:, 1] ** 2 + lb[:, 0] ** 2 + lb[:, 1] ** 2
        return r / np.sqrt(np.maximum(den, 1e-18))

    def _planar_check(self, result: TurnEstimate, xa, xb, thr: float, lo: float, hi: float) -> None:
        """General planar motion (yaw + translation direction alpha in base_link B):
        robust 2-parameter fit seeded from the turn estimate. A forward/backward
        base slide near a wall breaks the pure-turn model badly (several degrees)
        but is well explained here; lateral slides are nearly indistinguishable
        from the turn and bias it little."""
        cfg = self.cfg
        theta0 = result.theta if np.isfinite(result.theta) else 0.5 * (lo + hi)
        phi = theta0 / 2.0
        a_turn = math.atan2(-math.cos(phi), -math.sin(phi))  # pure-turn translation direction
        best = None
        for k in range(3):  # alpha and alpha + pi give the same constraint
            r = least_squares(self._planar_residuals, [theta0, a_turn + k * math.pi / 3],
                              args=(xa, xb), loss="cauchy", f_scale=cfg.cauchy_scale_px / self.fx)
            if best is None or r.cost < best.cost:
                best = r
        inl = np.abs(self._planar_residuals(best.x, xa, xb)) < thr
        gain = int(inl.sum()) - result.n_inliers
        if gain <= max(cfg.translation_gain_abs, cfg.translation_gain_frac * result.n_inliers):
            return
        xa_i, xb_i = xa[inl], xb[inl]
        res = self._planar_residuals(best.x, xa_i, xb_i)
        h = 1e-6
        J = np.column_stack([(self._planar_residuals(best.x + e, xa_i, xb_i)
                              - self._planar_residuals(best.x - e, xa_i, xb_i)) / (2 * h)
                             for e in (np.array([h, 0.0]), np.array([0.0, h]))])
        sigma = float("inf")
        if inl.sum() > 2:
            try:
                cov = np.linalg.inv(J.T @ J) * float((res ** 2).sum() / (inl.sum() - 2))
                sigma = math.sqrt(max(cov[0, 0], 0.0))
            except np.linalg.LinAlgError:
                pass
        result.translation_detected = True
        result.model = "planar"
        result.theta = float(best.x[0])
        result.sigma = sigma
        result.inliers = inl
        result.n_inliers = k = int(inl.sum())
        result.rms_px = float(math.sqrt((res ** 2).mean()) * self.fx)
        if k < cfg.min_inliers:
            result.ok, result.reason = False, "too few inliers (planar)"
        elif k < cfg.min_inlier_ratio * len(xa):
            result.ok, result.reason = False, "low inlier ratio (planar)"
        elif not np.isfinite(sigma) or sigma > cfg.max_sigma:
            result.ok, result.reason = False, "flat cost (planar)"
        elif not lo <= result.theta <= hi:
            result.ok, result.reason = False, "planar estimate outside window"
        else:
            result.ok, result.reason = True, ""


def estimate_turn(uv_a, uv_b, model, theta_hint: float | None = None,
                  max_abs: float = 1.0, threshold_px: float = 1.5) -> TurnEstimate:
    """One-shot convenience wrapper around :class:`TurnEpipolar`."""
    cfg = TurnEpipolarConfig(threshold_px=threshold_px, max_abs=max_abs)
    return TurnEpipolar(model, cfg).estimate(uv_a, uv_b, theta_hint)


_BF = None


def match_orb(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.8,
              max_hamming: int = 50) -> np.ndarray:
    """Hamming brute-force matching of ORB descriptors with Lowe ratio test.
    Returns an (M, 2) int array of (index_in_a, index_in_b)."""
    global _BF
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return np.empty((0, 2), int)
    if _BF is None:
        _BF = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = []
    for m in _BF.knnMatch(np.ascontiguousarray(desc_a, np.uint8), np.ascontiguousarray(desc_b, np.uint8), k=2):
        if len(m) == 2 and m[0].distance <= max_hamming and m[0].distance < ratio * m[1].distance:
            pairs.append((m[0].queryIdx, m[0].trainIdx))
    return np.array(pairs, int).reshape(-1, 2)
