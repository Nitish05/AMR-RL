"""Ground-truth scoring for evaluation ONLY.

Nothing in amr_rl.perception/mapping/navigation/learning/behavior/runtime may
import this module (enforced by tests/test_privilege_boundary.py). It reads the
simulator's true state to score what the robot estimated.
"""

from __future__ import annotations

import math

import numpy as np

from ..perception.camera_model import planar_T, wrap


def true_pose(world):
    pos = np.asarray(world.robot.get_pos()).reshape(3)
    w, x, y, z = np.asarray(world.robot.get_quat()).reshape(4)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([pos[0], pos[1], yaw])


def to_map_frame(origin_pose, pose):
    """Express a true world pose in the robot's map frame (origin = map start pose)."""
    T = np.linalg.inv(planar_T(*origin_pose)) @ planar_T(*pose)
    return np.array([T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])])


def map_to_world(origin_pose, xy):
    T = planar_T(*origin_pose)
    p = T @ np.array([xy[0], xy[1], 0.0, 1.0])
    return p[:2]


def world_to_map(origin_pose, xy):
    T = np.linalg.inv(planar_T(*origin_pose))
    p = T @ np.array([xy[0], xy[1], 0.0, 1.0])
    return p[:2]


def pose_error(estimate, truth):
    return float(np.hypot(estimate[0] - truth[0], estimate[1] - truth[1])), abs(wrap(estimate[2] - truth[2]))


def robot_contacts(world):
    """Names of scene entities currently touching the robot (floor excluded)."""
    try:
        c = world.robot.get_contacts()
    except Exception:  # noqa: BLE001
        return []
    ga = np.asarray(c.get("geom_a", [])).ravel()
    gb = np.asarray(c.get("geom_b", [])).ravel()
    r0, r1 = world.robot.geom_start, world.robot.geom_end
    solver = world.scene.rigid_solver
    names = set()
    for a, b in zip(ga, gb):
        other = int(b) if r0 <= a < r1 else int(a)
        if r0 <= other < r1:
            continue
        entity = solver.geoms[other].entity
        if entity.idx == world.plane.idx:
            continue
        names.add(world.entity_names.get(entity.idx, f"entity{entity.idx}"))
    return sorted(names)


def fixture_true_xy(world, name):
    item, entity = world.fixtures[name]
    return np.asarray(entity.get_pos()).reshape(3)[:2]


def true_obstacle_mask(world, grid, origin_pose, *, include_fixtures=True, margin=0.0):
    """Boolean [iy, ix] mask of grid cells whose centres (map frame) lie inside true
    static geometry footprints (walls, obstacles, fixtures) or outside the room."""
    n = grid.n
    iy, ix = np.mgrid[0:n, 0:n]
    centres = grid.to_xy(ix.ravel(), iy.ravel())
    T = planar_T(*origin_pose)
    world_xy = centres @ T[:2, :2].T + T[:2, 3]
    occupied = np.zeros(len(world_xy), bool)
    boxes = list(world.static_geometry)
    if include_fixtures:
        for name, (item, entity) in world.fixtures.items():
            xy = np.asarray(entity.get_pos()).reshape(3)[:2]
            size = item["size"]
            boxes.append({"name": name, "size": size, "xy": xy, "yaw": float(item.get("yaw", 0.0)),
                          "round": item["shape"] in ("cylinder", "sphere")})
    for box in boxes:
        d = world_xy - np.asarray(box["xy"])[None]
        if box.get("round"):
            occupied |= np.linalg.norm(d, axis=1) <= box["size"][0] / 2 + margin
            continue
        c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
        occupied |= (np.abs(local[:, 0]) <= box["size"][0] / 2 + margin) & (
            np.abs(local[:, 1]) <= box["size"][1] / 2 + margin)
    lx, ly = world.config["room"]["size"]
    occupied |= (np.abs(world_xy[:, 0]) > lx / 2) | (np.abs(world_xy[:, 1]) > ly / 2)
    return occupied.reshape(n, n)


def score_map(world, grid, origin_pose):
    from ..mapping.occupancy import FREE, OCCUPIED

    cls = grid.classes()
    truth = true_obstacle_mask(world, grid, origin_pose)
    # Cells more than one cell inside true geometry: unambiguous violations.
    import cv2

    deep = cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    free = cls == FREE
    return {
        "free_cells": int(free.sum()),
        "occupied_cells": int((cls == OCCUPIED).sum()),
        "false_free_cells": int((free & truth).sum()),
        "false_free_deep_cells": int((free & deep).sum()),
        "true_free_cells_in_room": int((~truth).sum()),
        "free_coverage": float((free & ~truth).sum() / max(1, (~truth).sum())),
        "occupied_on_truth": int(((cls == OCCUPIED) & truth).sum()),
    }


def score_guard(world, grid, origin_pose, asserted_log, margin=0.1):
    """Scoring only: where did the near-field guard assert fresh obstacles? Cells within
    ``margin`` of true geometry count as justified; the rest lie on open floor."""
    cells = set()
    for _, xy in asserted_log:
        if len(xy):
            ix, iy = grid.to_cell(np.asarray(xy, float))
            cells |= set(zip(ix.tolist(), iy.tolist()))
    if not cells:
        return {"asserted_cells": 0, "near_true_geometry": 0, "on_open_floor": 0}
    near = true_obstacle_mask(world, grid, origin_pose, margin=margin)
    on = sum(bool(near[y, x]) for x, y in cells if 0 <= x < grid.n and 0 <= y < grid.n)
    return {"asserted_cells": len(cells), "near_true_geometry": int(on), "on_open_floor": len(cells) - int(on)}
