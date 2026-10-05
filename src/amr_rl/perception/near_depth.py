"""Near-field obstacles from one RGB frame with a monocular depth model.

Camera-only contract: pixels in, obstacle points in the robot's base frame out. No
simulator state, no motor commands. The model is optional: it is loaded only from
the local Hugging Face cache (``scripts/amr.sh fetch-depth-model`` downloads it,
outside the repository), and if it is missing the caller simply has no detector.

Model: Depth Anything V2 Small (24.8 M parameters, Apache-2.0), pinned revision.
It predicts *relative* inverse depth (unknown scale and shift). Metric depth is
recovered per frame from the robot's own calibration: below the horizon, pixels
that are floor must have inverse depth 1/z_floor(u, v) given the camera height and
pitch, so a robust affine fit ``model = a / z + b`` over the lower image fixes a and
b. Pixels that then come out noticeably above the floor (``min_height``) within
``max_range`` are obstacle points. Something standing on the floor fails the floor
fit where it is, whatever it looks like.

Floor mask (``MonoDepthObstacles.floor_mask`` / ``mask_points``): the same model
output tells the VSLAM which image pixels are *not* floor, so features on a near
wall or box face are not lifted to the floor by inverse perspective mapping (they
would land far too distant). For a flat floor and a camera without roll, the metric
inverse depth of the floor along each pixel ray is exact from calibration,
``1/z_floor = -(up_cam . ray) / camera_height`` (linear in the image row). The model
disparity is fitted as ``d = a / z_floor + b`` by RANSAC on a bottom floor band,
smoothed across calls, held (and flagged stale) when too little floor is visible; a
pixel is not floor when the model puts it clearly nearer than the floor would be
(``z_model < depth_ratio * z_floor``).

Engineered: thresholds, fit region, input size, the floor-mask fit and threshold.
Learned (by the model's authors, not by this robot): the depth prior.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_LICENSE = "apache-2.0"


@dataclass
class DepthConfig:
    input_width: int = 336        # model input width (multiple of 14); ~0.2 s on 2 CPU cores
    horizon_margin_px: float = 6  # fit and detect only this far below the horizon
    fit_max_range: float = 2.0    # m; floor pixels used for the scale/shift fit
    fit_keep: float = 0.6         # trimmed least squares: keep this fraction of residuals
    fit_iterations: int = 4
    min_height: float = 0.05      # m above the floor to count as an obstacle point
    max_height: float = 0.8
    max_range: float = 1.2        # m from the base origin
    stride: int = 2               # pixel subsampling for detection
    min_fit_pixels: int = 400


@dataclass
class FloorMaskConfig:
    band_top_frac: float = 0.75   # fit band: image rows v >= band_top_frac * height (bottom 25 %)
    fit_stride: int = 2           # pixel subsampling of the fit band
    ransac_iters: int = 96
    inlier_rel: float = 0.05      # inlier: |d - (a x + b)| <= inlier_rel * (a x + b)
    min_inliers: int = 300        # (subsampled) floor inliers needed for a fresh fit
    min_span: float = 0.5         # inliers must cover this fraction of the band's floor inverse-depth range
    min_contrast: float = 5.0     # fitted floor disparity range over the band >= min_contrast x tolerance
    refine_iterations: int = 3
    refine_keep: float = 0.6      # trimmed least-squares refinement keeps this fraction of inliers
    ema_alpha: float = 0.2        # weight of a fresh fit in the running (a, b)
    ema_reset_rel: float = 0.05   # fresh fit disagreeing with the running fit by more restarts it
    depth_ratio: float = 0.85     # not floor where model depth < depth_ratio * IPM floor depth
    horizon_margin_px: float = 0.0  # judge pixels this far below the horizon (above: not floor)
    max_floor_depth: float = 3.0  # m; pixels whose floor is farther are not judged (left False):
                                  # the band fit extrapolated that far is too uncertain
    seed: int = 0                 # RANSAC sampling is deterministic per call


def resolve_device(device: str = "cpu") -> str:
    """'cpu' | 'mps' | 'cuda' | 'auto' (mps, then cuda, then cpu) -> torch device name."""
    if device not in ("cpu", "mps", "cuda", "auto"):
        raise ValueError(f"unknown depth device {device!r} (cpu | mps | cuda | auto)")
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if device == "auto":
        return "mps" if mps else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not mps or device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"depth device {device!r} is not available")
    return device


def load_backend(input_width: int = 336, *, allow_download: bool = False, device: str = "cpu"):
    """Return ``fn(rgb uint8 HxWx3) -> relative inverse depth HxW`` or None if the
    model (or torch/transformers) is unavailable. Never downloads unless asked.
    ``device``: 'cpu' (default; deterministic for tests and benches), 'mps', 'cuda' or
    'auto'. The returned function carries the resolved device as ``fn.device``."""
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
    except ImportError:
        return None
    dev = resolve_device(device)
    try:
        model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=not allow_download).eval()
    except Exception:  # not in the local cache
        return None
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    if dev != "cpu":
        model, mean, std = model.to(dev), mean.to(dev), std.to(dev)

    def infer(rgb):
        h, w = rgb.shape[:2]
        iw = input_width // 14 * 14
        ih = max(14, int(round(input_width * h / w / 14)) * 14)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        if dev != "cpu":
            x = x.to(dev)
        x = torch.nn.functional.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            out = model(pixel_values=x).predicted_depth[:, None]
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out[0, 0].cpu().numpy().astype(np.float32)

    infer.device = dev
    return infer


def model_available() -> bool:
    """Cheap check (no model load, no network): are torch, transformers and the pinned
    weights present locally?"""
    import importlib.util

    if any(importlib.util.find_spec(m) is None for m in ("torch", "transformers", "huggingface_hub")):
        return False
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(MODEL_ID, f, revision=MODEL_REVISION), str)
               for f in ("config.json", "model.safetensors"))


class LazyBackend:
    """Loads the model on first use, so constructing a runtime stays fast."""

    def __init__(self, input_width: int = 336, device: str = "cpu"):
        self.input_width = input_width
        self.requested_device = device
        self._fn = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        """Resolved device once loaded, else the requested one."""
        return getattr(self._fn, "device", self.requested_device)

    def __call__(self, rgb):
        if self._fn is None:
            with self._lock:
                if self._fn is None:
                    fn = load_backend(self.input_width, device=self.requested_device)
                    if fn is None:
                        raise RuntimeError("depth model could not be loaded from the local cache")
                    self._fn = fn
        return self._fn(rgb)


class MonoDepthObstacles:
    def __init__(self, camera, config: DepthConfig | None = None, backend=None,
                 floor_config: FloorMaskConfig | None = None):
        self.cam = camera
        self.cfg = config or DepthConfig()
        self.floor_cfg = floor_config or FloorMaskConfig()
        self.backend = backend
        h, w = camera.height, camera.width
        v, u = np.mgrid[0:h, 0:w]
        self.uv = np.stack([u.ravel(), v.ravel()], 1).astype(float)
        self.rays = camera.rays(self.uv)  # camera frame, z = 1
        T = camera.T_base_cam
        self.R, self.t = T[:3, :3], T[:3, 3]
        # Floor geometry at base pose (0, 0, 0): camera-frame depth of the floor point.
        ground, ok = camera.ground_points_world(self.uv, (0.0, 0.0, 0.0), max_range=10.0)
        below = self.uv[:, 1] > camera.horizon_row() + self.cfg.horizon_margin_px
        self.floor_ok = ok & below
        Tc = np.linalg.inv(camera.T_world_cam((0.0, 0.0, 0.0)))
        pc = ground @ Tc[:3, :3].T + Tc[:3, 3]
        self.z_floor = np.where(self.floor_ok, pc[:, 2], np.nan)
        rng = np.linalg.norm(ground[:, :2], axis=1)
        self.fit_mask = self.floor_ok & (rng <= self.cfg.fit_max_range)
        sub = np.zeros((h, w), bool)
        sub[::self.cfg.stride, ::self.cfg.stride] = True
        self.detect_mask = below & sub.ravel()
        self.stats = {"frames": 0, "fit_failed": 0}
        self._lock = threading.RLock()
        self._infer_key = None   # one-frame cache shared by detect() and the floor mask
        self._infer_out = None
        self.inferences = 0      # model calls actually made (cache misses)
        self._init_floor_mask()

    @property
    def available(self):
        return self.backend is not None

    def fit(self, rel):
        """Robust affine fit rel = a / z_floor + b over floor-candidate pixels."""
        y = rel.ravel()[self.fit_mask]
        x = 1.0 / self.z_floor[self.fit_mask]
        keep = np.isfinite(y) & np.isfinite(x)
        x, y = x[keep], y[keep]
        if len(x) < self.cfg.min_fit_pixels:
            return None
        A = np.column_stack([x, np.ones_like(x)])
        sel = np.ones(len(x), bool)
        for _ in range(self.cfg.fit_iterations):
            coef, *_ = np.linalg.lstsq(A[sel], y[sel], rcond=None)
            res = np.abs(A @ coef - y)
            sel = res <= np.quantile(res, self.cfg.fit_keep)
        a, b = float(coef[0]), float(coef[1])
        if a <= 0:
            return None
        spread = float(np.median(res[sel]) / max(np.ptp(y[sel]), 1e-6))
        return a, b, spread

    def detect(self, rgb, rel=None):
        """Obstacle points (N, 2) in the base frame (x forward, y left), with heights and
        the fit. ``rel`` may be passed in (precomputed model output)."""
        self.stats["frames"] += 1
        if rel is None:
            if self.backend is None:
                return None
            rel = self.infer(rgb)
        fit = self.fit(rel)
        if fit is None:
            self.stats["fit_failed"] += 1
            return {"points": np.zeros((0, 2)), "heights": np.zeros(0), "fit": None}
        a, b, spread = fit
        d = rel.ravel()[self.detect_mask] - b
        good = d > 1e-6
        z = np.full(d.shape, np.nan)
        z[good] = a / d[good]
        pc = self.rays[self.detect_mask] * z[:, None]
        pb = pc @ self.R.T + self.t
        height = pb[:, 2] + self.cam.base_z
        rng = np.linalg.norm(pb[:, :2], axis=1)
        obst = good & (height >= self.cfg.min_height) & (height <= self.cfg.max_height) & \
            (rng <= self.cfg.max_range) & (pb[:, 0] > 0)
        return {"points": pb[obst, :2], "heights": height[obst], "fit": {"a": a, "b": b, "spread": spread}}

    # ------------------------------------------------------------ shared inference
    @staticmethod
    def frame_key(img):
        """Identity of one frame: the array object plus a hash of a sparse sample of its
        pixels (catches camera buffers reused in place)."""
        img = np.asarray(img)
        sample = np.ascontiguousarray(img[::8, ::8])
        return (id(img), img.shape, str(img.dtype), hashlib.blake2b(sample.tobytes(), digest_size=16).digest())

    def infer(self, rgb):
        """Model output (relative inverse depth, camera image size HxW, read-only) for
        one frame. The last frame is cached, so the obstacle guard and the floor mask
        share one inference. A grey image is replicated to three channels (the model
        expects RGB; prefer passing the colour frame)."""
        if self.backend is None:
            raise RuntimeError("no depth backend")
        key = self.frame_key(rgb)
        with self._lock:
            if self._infer_key == key:
                return self._infer_out
            img = np.asarray(rgb)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            rel = np.asarray(self.backend(img), np.float32)
            rel = self._to_camera_size(rel)
            rel.flags.writeable = False
            self._infer_key, self._infer_out = key, rel
            self.inferences += 1
            return rel

    def _to_camera_size(self, rel):
        h, w = self.cam.height, self.cam.width
        if rel.shape == (h, w):
            return rel
        rows = np.minimum(((np.arange(h) + 0.5) * rel.shape[0] / h).astype(int), rel.shape[0] - 1)
        cols = np.minimum(((np.arange(w) + 0.5) * rel.shape[1] / w).astype(int), rel.shape[1] - 1)
        return np.ascontiguousarray(rel[rows][:, cols])

    # ------------------------------------------------------------ floor mask
    def _init_floor_mask(self):
        fc, cam = self.floor_cfg, self.cam
        h, w = cam.height, cam.width
        # Exact IPM floor inverse depth along each pixel ray (camera z = 1 rays):
        # a floor point satisfies up_cam . (z * ray) = -camera_height.
        up_cam = self.R.T @ np.array([0.0, 0.0, 1.0])
        inv = -(self.rays @ up_cam) / cam.camera_height
        v = self.uv[:, 1]
        below = (inv > 1e-6) & (v > cam.horizon_row() + fc.horizon_margin_px)
        self.floor_eval = below & (inv * fc.max_floor_depth >= 1.0)   # pixels that are judged
        self.inv_floor = np.where(below, inv, np.nan)
        self.never_floor = ~below                                    # at or above the horizon
        sub = np.zeros((h, w), bool)
        sub[::fc.fit_stride, ::fc.fit_stride] = True
        band = self.floor_eval & (v >= fc.band_top_frac * h) & sub.ravel()
        self.band_idx = np.flatnonzero(band)
        self.band_x = inv[self.band_idx]
        self.band_span = float(np.ptp(self.band_x)) if len(self.band_x) else 0.0
        self.band_lo = float(self.band_x.min()) if len(self.band_x) else 0.0
        self.band_hi = float(self.band_x.max()) if len(self.band_x) else 0.0
        self.floor_fit = None       # running (a, b): model disparity = a / z_floor + b
        self.floor_diag = {"valid": False, "stale": True, "stale_frames": 0, "frames": 0}
        self._floor_key = None
        self._floor_out = None

    def _plausible(self, a, b):
        """Positive slope, positive disparity over the band, and a fitted floor that
        varies by clearly more than the inlier tolerance across the band (rejects a
        near-constant fit to a wall face filling the band)."""
        fc = self.floor_cfg
        f_lo, f_hi = a * self.band_lo + b, a * self.band_hi + b
        f_mid = 0.5 * (f_lo + f_hi)
        return (a > 0) & (f_lo > 0) & (a * self.band_span >= fc.min_contrast * fc.inlier_rel * f_mid)

    def fit_floor_band(self, rel):
        """RANSAC fit of ``rel = a * inv_floor + b`` on the bottom floor band of this
        frame. Returns ``(a, b, n_inliers, span_fraction)`` or None and the inlier count."""
        fc = self.floor_cfg
        y = np.asarray(rel, np.float64).ravel()[self.band_idx]
        x = self.band_x
        fin = np.isfinite(y)
        x, y = x[fin], y[fin]
        n = len(x)
        if n < fc.min_inliers or self.band_span <= 0:
            return None, 0
        rng = np.random.default_rng(fc.seed)
        i, j = rng.integers(0, n, fc.ransac_iters), rng.integers(0, n, fc.ransac_iters)
        dx = x[j] - x[i]
        use = np.abs(dx) > 0.1 * self.band_span
        if not use.any():
            return None, 0
        i, j, dx = i[use], j[use], dx[use]
        a = (y[j] - y[i]) / dx
        b = y[i] - a * x[i]
        good = self._plausible(a, b)
        if not good.any():
            return None, 0
        a, b = a[good], b[good]
        f = a[:, None] * x[None, :] + b[:, None]
        counts = (np.abs(y[None, :] - f) <= fc.inlier_rel * np.abs(f)).sum(1)
        k = int(np.argmax(counts))
        ak, bk = float(a[k]), float(b[k])
        inl = np.abs(y - (ak * x + bk)) <= fc.inlier_rel * np.abs(ak * x + bk)
        for _ in range(fc.refine_iterations):
            # Trimmed least squares on the consensus set: the best-fitting refine_keep of
            # the inliers, so pixels that only just pass (e.g. the foot of a box face,
            # where it meets the floor) do not bias the slope and offset.
            r = np.abs(y - (ak * x + bk)) / np.maximum(np.abs(ak * x + bk), 1e-9)
            use = inl & (r <= np.quantile(r[inl], fc.refine_keep)) if inl.any() else inl
            if use.sum() < 2 or np.ptp(x[use]) <= 0:
                break
            ak, bk = (float(c) for c in np.polyfit(x[use], y[use], 1))
            inl = np.abs(y - (ak * x + bk)) <= fc.inlier_rel * np.abs(ak * x + bk)
        n_in = int(inl.sum())
        if n_in < fc.min_inliers or not self._plausible(ak, bk):
            return None, n_in
        lo, hi = np.quantile(x[inl], [0.05, 0.95])
        span = float((hi - lo) / self.band_span)
        if span < fc.min_span:
            return None, n_in
        return (ak, bk, n_in, span), n_in

    def _fit_disagreement(self, f1, f2):
        """Largest relative difference of two (a, b) floor fits over the band."""
        out = 0.0
        for x in (self.band_lo, self.band_hi):
            p, q = f1[0] * x + f1[1], f2[0] * x + f2[1]
            out = max(out, abs(p - q) / max(abs(p), 1e-9))
        return out

    def floor_mask(self, rgb, rel=None):
        """``(not_floor HxW bool, diagnostics)`` for one frame. True = not floor: at or
        above the horizon, or below it where the model puts the pixel clearly nearer than
        the floor along that ray (``z_model < depth_ratio * z_floor``). Pixels whose floor
        would be beyond ``max_floor_depth`` are not judged (False). Without any floor fit
        yet nothing below the horizon is masked (``diag['valid']`` False).

        Calls are cached per frame (``frame_key``) so the running fit is updated once
        per frame; ``rel`` (precomputed model output) bypasses inference and cache."""
        with self._lock:
            key = None
            if rel is None:
                key = self.frame_key(rgb)
                if self._floor_key == key:
                    return self._floor_out
                rel = self.infer(rgb)
            else:
                rel = self._to_camera_size(np.asarray(rel, np.float32))
            out = self._floor_mask(rel)
            if key is not None:
                self._floor_key, self._floor_out = key, out
            return out

    def _floor_mask(self, rel):
        fc = self.floor_cfg
        h, w = self.cam.height, self.cam.width
        fresh, n_in = self.fit_floor_band(rel)
        prev = self.floor_diag
        reset = False
        if fresh is not None:
            new = (fresh[0], fresh[1])
            if self.floor_fit is None or self._fit_disagreement(new, self.floor_fit) > fc.ema_reset_rel:
                reset = self.floor_fit is not None
                self.floor_fit = new
            else:
                al = fc.ema_alpha
                self.floor_fit = ((1 - al) * self.floor_fit[0] + al * new[0],
                                  (1 - al) * self.floor_fit[1] + al * new[1])
        stale = fresh is None
        diag = {
            "valid": self.floor_fit is not None,
            "fresh": not stale,
            "stale": stale,
            "stale_frames": (prev.get("stale_frames", 0) + 1) if stale else 0,
            "frames": prev.get("frames", 0) + 1,
            "inliers": int(n_in),
            "band_pixels": len(self.band_idx),
            "span": fresh[3] if fresh is not None else None,
            "a": None, "b": None,
            "a_frame": fresh[0] if fresh is not None else None,
            "b_frame": fresh[1] if fresh is not None else None,
            "ema_reset": reset,
            "masked_fraction": 0.0,
        }
        not_floor = self.never_floor.copy()
        if self.floor_fit is not None:
            a, b = self.floor_fit
            diag["a"], diag["b"] = a, b
            inv_model = (np.asarray(rel, np.float64).ravel() - b) / a
            with np.errstate(invalid="ignore"):
                near = self.floor_eval & (inv_model * fc.depth_ratio > self.inv_floor)
            not_floor = not_floor | near
            diag["masked_fraction"] = float(near.sum() / max(int(self.floor_eval.sum()), 1))
        self.floor_diag = diag
        not_floor = not_floor.reshape(h, w)
        not_floor.flags.writeable = False  # may be returned again from the per-frame cache
        return not_floor, diag

    def mask_points(self, rgb, pts):
        """For pixel coordinates ``pts`` (N, 2; u, v) of this frame: True = not floor
        (nearest-pixel lookup in ``floor_mask``). The last diagnostics are in
        ``self.floor_diag``."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros(0, bool)
        mask, _ = self.floor_mask(rgb)
        h, w = mask.shape
        u = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        return mask[v, u]
