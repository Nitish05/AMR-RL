"""Salient-object detector for the engineered fixture set (onboard RGB only).

Scope, stated plainly: this finds uniformly painted, strongly saturated regions
(the rooms are deliberately desaturated) and treats a saturated region stacked
directly on top of another as an *attachment* of the lower object (its visible
state). It is an engineered-fixture detector, not general object recognition.
Positions come from the object's floor contact row via the calibrated camera
height and the VSLAM pose; no simulator state is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

HUE_NAMES = [(15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "cyan"),
             (250, "blue"), (290, "violet"), (345, "magenta"), (361, "red")]


def hue_name(hue_deg: float) -> str:
    for limit, name in HUE_NAMES:
        if hue_deg < limit:
            return name
    return "red"


def _rests_on(upper, lower, lo, hi, max_gap=8, min_fraction=0.5):
    """True if, in most shared columns where ``upper`` has pixels, ``lower``'s first
    pixel below ``upper``'s lowest pixel starts within ``max_gap`` rows."""
    cols = up = 0
    for c in range(lo, hi):
        ua = np.flatnonzero(upper[:, c])
        if ua.size == 0:
            continue
        cols += 1
        below = np.flatnonzero(lower[ua[-1] + 1:, c])
        if below.size and below[0] < max_gap:
            up += 1
    return cols > 0 and up >= min_fraction * cols


def circular_mean_deg(h):
    a = np.radians(np.asarray(h, float))
    return float(np.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def hue_distance(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


@dataclass
class Attachment:
    hue: float
    color: str
    area: int
    bbox: tuple


@dataclass
class Detection:
    index: int
    bbox: tuple  # x0, y0, x1, y1 (inclusive-exclusive)
    area: int
    hue: float
    color: str
    saturation: float
    fill: float
    base_uv: tuple
    partial: bool
    attachments: list = field(default_factory=list)
    position: tuple | None = None  # estimated map-frame centre (x, y)
    range_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    bearing: float | None = None  # radians, CCW-positive in the robot frame

    @property
    def state_token(self) -> str:
        """Visible attachment state used as interaction context/outcome evidence."""
        if not self.attachments:
            return "attach:none"
        main = max(self.attachments, key=lambda a: a.area)
        return f"attach:{main.color}"

    def descriptor(self):
        return {"hue": self.hue, "color": self.color, "saturation": self.saturation, "fill": self.fill,
                "width_m": self.width_m, "height_m": self.height_m}


class FixtureDetector:
    def __init__(self, model, *, min_saturation=0.5, min_value=0.18, min_area=40):
        self.model = model
        self.min_s = int(255 * min_saturation)
        self.min_v = int(255 * min_value)
        self.min_area = min_area
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, rgb: np.ndarray, pose=None) -> list[Detection]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] >= self.min_s) & (hsv[..., 2] >= self.min_v)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        hue = hsv[..., 0].astype(np.float32) * 2.0
        # Split touching regions of clearly different hue: label per hue family.
        family = np.zeros(mask.shape, np.int32)
        centers = np.array([0, 55, 130, 185, 225, 270, 320], float)
        d = np.abs(((hue[..., None] - centers[None, None]) + 180) % 360 - 180)
        family = d.argmin(-1) + 1
        family[mask == 0] = 0
        blobs = []
        for fam in range(1, len(centers) + 1):
            m = (family == fam).astype(np.uint8)
            if m.sum() < self.min_area:
                continue
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            for k in range(1, n):
                x, y, w, h, area = stats[k]
                if area < self.min_area:
                    continue
                region = labels == k
                blobs.append({
                    "bbox": (int(x), int(y), int(x + w), int(y + h)), "area": int(area),
                    "hue": circular_mean_deg(hue[region]),
                    "sat": float(hsv[..., 1][region].mean() / 255.0),
                    "fill": float(area / max(1, w * h)),
                    "region": region,
                })
        # Attachment rule: a smaller blob whose bottom edge rests on another blob's top
        # edge, tested column by column (bounding boxes can overlap when the body's
        # rounded top edge occludes the bottom of the attachment at close range).
        attached = set()
        parents = {}
        for i, a in enumerate(blobs):
            ax0, ay0, ax1, ay1 = a["bbox"]
            for j, b in enumerate(blobs):
                if i == j or a["area"] >= b["area"]:
                    continue
                bx0, by0, bx1, by1 = b["bbox"]
                lo, hi = max(ax0, bx0), min(ax1, bx1)
                if hi - lo <= 0.5 * (ax1 - ax0) or by0 > ay1 + 8 or ay1 > by1:
                    continue
                if _rests_on(a["region"], b["region"], lo, hi):
                    attached.add(i)
                    parents[i] = j
        detections = []
        H, W = mask.shape
        for j, b in enumerate(blobs):
            if j in attached:
                continue
            x0, y0, x1, y1 = b["bbox"]
            ys, xs = np.nonzero(b["region"])
            bottom = ys.max()
            base_u = float(xs[ys >= bottom - 1].mean())
            det = Detection(
                index=len(detections), bbox=b["bbox"], area=b["area"], hue=b["hue"], color=hue_name(b["hue"]),
                saturation=b["sat"], fill=b["fill"], base_uv=(base_u, float(bottom)),
                partial=bool(x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1),
            )
            for i, parent in parents.items():
                if parent == j:
                    a = blobs[i]
                    det.attachments.append(Attachment(a["hue"], hue_name(a["hue"]), a["area"], a["bbox"]))
            self._localize(det, pose)
            detections.append(det)
        return detections

    def _localize(self, det: Detection, pose):
        if det.partial and det.bbox[3] >= self.model.height - 1:
            return  # floor contact not visible
        # contact point in the robot's base frame (pose 0) then map frame
        base_pose = (0.0, 0.0, 0.0)
        P, ok = self.model.ground_points_world(np.array([det.base_uv]), base_pose, max_range=5.0)
        if not ok[0]:
            return
        cam = self.model.T_world_cam(base_pose)[:3, 3]
        contact = P[0]
        slant = float(np.linalg.norm(contact - cam))
        f = self.model.K[0, 0]
        x0, y0, x1, y1 = det.bbox
        det.width_m = (x1 - x0) * slant / f
        det.height_m = (y1 - y0) * slant / self.model.K[1, 1]
        ray = contact[:2] - cam[:2]
        ray /= max(np.linalg.norm(ray), 1e-6)
        # A round silhouette's lowest point lies under its centre; a box/cylinder's
        # lowest visible edge is its front face (engineered shape heuristic).
        offset = 0.0 if det.fill < 0.83 else det.width_m / 2
        centre_base = contact[:2] + ray * offset
        det.range_m = float(np.linalg.norm(centre_base))
        det.bearing = math.atan2(centre_base[1], centre_base[0])
        if pose is None:
            det.position = None
            return
        c, s = math.cos(pose[2]), math.sin(pose[2])
        det.position = (pose[0] + c * centre_base[0] - s * centre_base[1],
                        pose[1] + s * centre_base[0] + c * centre_base[1])
