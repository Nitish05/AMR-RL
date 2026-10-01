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

Engineered: thresholds, fit region, input size. Learned (by the model's authors,
not by this robot): the depth prior.
"""

from __future__ import annotations

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


def load_backend(input_width: int = 336, *, allow_download: bool = False):
    """Return ``fn(rgb uint8 HxWx3) -> relative inverse depth HxW`` or None if the
    model (or torch/transformers) is unavailable. Never downloads unless asked."""
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
    except ImportError:
        return None
    try:
        model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=not allow_download).eval()
    except Exception:  # not in the local cache
        return None
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def infer(rgb):
        h, w = rgb.shape[:2]
        iw = input_width // 14 * 14
        ih = max(14, int(round(input_width * h / w / 14)) * 14)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        x = torch.nn.functional.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            out = model(pixel_values=x).predicted_depth[:, None]
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out[0, 0].numpy().astype(np.float32)

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

    def __init__(self, input_width: int = 336):
        self.input_width = input_width
        self._fn = None

    def __call__(self, rgb):
        if self._fn is None:
            self._fn = load_backend(self.input_width)
            if self._fn is None:
                raise RuntimeError("depth model could not be loaded from the local cache")
        return self._fn(rgb)


class MonoDepthObstacles:
    def __init__(self, camera, config: DepthConfig | None = None, backend=None):
        self.cam = camera
        self.cfg = config or DepthConfig()
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
            rel = self.backend(rgb)
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
