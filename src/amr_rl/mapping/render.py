"""Render the estimated occupancy map (north-up) for the operator console."""

from __future__ import annotations

import cv2
import numpy as np

from .occupancy import FREE, OCCUPIED

COLORS = {"unknown": (38, 42, 50), "free": (214, 222, 228), "occupied": (40, 40, 44)}


def render_map(grid, *, trajectory=(), pose=None, sigma=None, path=(), goal=None, entities=(),
               scale=4, crop=None, footprint_radius=0.26):
    """Return (rgb, meta). ``meta`` maps image pixels to map coordinates:
    x = origin[0] + u * resolution, y = origin[1] + (height_px - v) * resolution."""
    cls = grid.classes()
    ix0, iy0, ix1, iy1 = 0, 0, grid.n, grid.n
    if crop is None:
        known = np.argwhere(cls != 0)
        if len(known):
            iy0, ix0 = np.maximum(known.min(0) - 12, 0)
            iy1, ix1 = np.minimum(known.max(0) + 13, grid.n)
    sub = cls[iy0:iy1, ix0:ix1]
    img = np.empty((*sub.shape, 3), np.uint8)
    img[:] = COLORS["unknown"]
    img[sub == FREE] = COLORS["free"]
    img[sub == OCCUPIED] = COLORS["occupied"]
    img = img[::-1]  # north-up: row 0 = max y
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    res = grid.cfg.resolution / scale
    origin = grid.origin + np.array([ix0, iy0]) * grid.cfg.resolution
    height = img.shape[0]

    def px(xy):
        return (int(round((xy[0] - origin[0]) / res)), int(round(height - (xy[1] - origin[1]) / res)))

    if len(trajectory) > 1:
        pts = np.array([px(p) for p in trajectory], np.int32)
        cv2.polylines(img, [pts], False, (90, 150, 255), 1, cv2.LINE_AA)
    if len(path) > 1:
        pts = np.array([px(p) for p in path], np.int32)
        cv2.polylines(img, [pts], False, (60, 200, 90), 2, cv2.LINE_AA)
    if goal is not None:
        cv2.drawMarker(img, px(goal), (40, 170, 60), cv2.MARKER_TILTED_CROSS, 12, 2)
    for entity in entities:
        if entity.get("position") is None:
            continue
        color = {"liked": (60, 190, 90), "disliked": (220, 60, 60), "indifferent": (150, 150, 150)}.get(
            entity.get("attitude"), (240, 200, 40))
        cv2.circle(img, px(entity["position"]), 6, color, 2, cv2.LINE_AA)
    if pose is not None:
        c = px(pose[:2])
        cv2.circle(img, c, max(2, int(footprint_radius / res)), (255, 140, 40), 1, cv2.LINE_AA)
        tip = px((pose[0] + 0.25 * np.cos(pose[2]), pose[1] + 0.25 * np.sin(pose[2])))
        cv2.arrowedLine(img, c, tip, (255, 120, 20), 2, cv2.LINE_AA, tipLength=0.35)
        if sigma:
            cv2.circle(img, c, max(1, int(sigma / res)), (255, 60, 60), 1, cv2.LINE_AA)
    meta = {"origin": [float(origin[0]), float(origin[1])], "resolution": float(res),
            "width_px": int(img.shape[1]), "height_px": int(img.shape[0])}
    return img, meta
