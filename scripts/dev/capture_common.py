"""Scoring geometry shared by the evaluation capture scripts (simulator truth; never
imported by the runtime)."""

from __future__ import annotations

import math

import numpy as np


def scene_boxes(world, *, walls=True):
    """True footprints: walls/furniture boxes and fixtures (round ones as discs)."""
    boxes = [dict(g) for g in world.static_geometry if g["name"] != "evaluation_obstacle"
             and (walls or not g["name"].startswith("wall_"))]
    for item, entity in world.fixtures.values():
        xy = np.asarray(entity.get_pos()).reshape(3)[:2]
        boxes.append({"name": item["name"], "size": item["size"], "xy": xy, "yaw": float(item.get("yaw", 0.0)),
                      "round": item["shape"] in ("cylinder", "sphere")})
    return boxes


def footprint_distance(p, box):
    """Distance from point p to a footprint (0 inside)."""
    d = np.asarray(p, float) - np.asarray(box["xy"], float)
    if box.get("round"):
        return max(0.0, float(np.linalg.norm(d)) - box["size"][0] / 2)
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    loc = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return float(np.linalg.norm(np.maximum(np.abs(loc) - np.asarray(box["size"][:2]) / 2, 0.0)))


def _closest_point(p, box):
    d = np.asarray(p, float) - np.asarray(box["xy"], float)
    if box.get("round"):
        n = np.linalg.norm(d)
        return np.asarray(box["xy"], float) + (d / max(n, 1e-9)) * box["size"][0] / 2
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    loc = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    half = np.asarray(box["size"][:2]) / 2
    q = np.clip(loc, -half, half)
    return np.asarray(box["xy"], float) + np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]])


def blocked(world, p, r):
    """Does a disc of radius r at p overlap walls, furniture or fixtures?"""
    lx, ly = world.config["room"]["size"]
    if abs(p[0]) > lx / 2 - r or abs(p[1]) > ly / 2 - r:
        return True
    return any(footprint_distance(p, b) <= r for b in scene_boxes(world, walls=False))


def nearest_object(world, p, heading=None, hfov_deg=42.5):
    """Distance (m, from the base point) to the nearest non-wall object footprint;
    with ``heading``, only objects whose closest point lies within +-hfov of it."""
    best = math.inf
    for b in scene_boxes(world, walls=False):
        if heading is not None:
            q = _closest_point(p, b) - np.asarray(p, float)
            bearing = math.atan2(q[1], q[0]) - heading
            if abs(math.atan2(math.sin(bearing), math.cos(bearing))) > math.radians(hfov_deg):
                continue
        best = min(best, footprint_distance(p, b))
    return best


def keyframe_distance(kf_world, p):
    return float(np.min(np.linalg.norm(np.asarray(kf_world) - np.asarray(p)[None], axis=1)))
