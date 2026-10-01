"""Nudge reach: does the planned creep reach the fixture? (evaluation probe)

The robot is placed at nudge standoff ranges and several approach angles, facing the
fixture. For each view, the creep planned from the detection (old: centre range minus
a radius; new: range to the nearest visible floor contact) is compared with the true
forward gap between the robot's front and the fixture surface along its heading.
Uses the true pose only to measure geometry; the runtime never sees it.

    PYTHONPATH=src python scripts/dev/nudge_reach_probe.py [world]
"""

import math
import sys

import numpy as np

from amr_rl.behavior.activities import FRONT_EXTENT, PUSH_DEPTH, nudge_creep
from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.entities import FixtureDetector
from amr_rl.sim import evaluator
from amr_rl.sim.world import SimWorld, load_world_config


def forward_gap(pose, item):
    """Distance along the heading from the robot's front to the fixture surface."""
    x, y, th = pose
    d = np.array([math.cos(th), math.sin(th)])
    fx, fy = item["pos"]
    sx, sy = float(item["size"][0]) / 2, float(item["size"][1]) / 2
    if item["shape"] in ("sphere", "cylinder"):
        c = np.array([fx - x, fy - y])
        along = float(c @ d)
        perp2 = float(c @ c) - along ** 2
        if perp2 > sx ** 2:
            return math.inf
        return along - math.sqrt(sx ** 2 - perp2) - FRONT_EXTENT
    # axis-aligned box (fixtures are placed with yaw 0): slab intersection
    lo = np.array([fx - sx - x, fy - sy - y])
    hi = np.array([fx + sx - x, fy + sy - y])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / d, hi / d
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    if tmax < max(tmin, 0):
        return math.inf
    return float(tmin) - FRONT_EXTENT


world = SimWorld(load_world_config(sys.argv[1] if len(sys.argv) > 1 else "arena"), inspection=False)
det = FixtureDetector(CameraModel.from_spec(world.spec))
rows = []
for name, (item, _) in world.fixtures.items():
    fx, fy = item["pos"]
    for rng in (0.47, 0.52, 0.58):
        for ang in (0, 20, 45, 70, 110, 160, 200, 250, 315):
            a = math.radians(ang)
            rx, ry = fx + rng * math.cos(a), fy + rng * math.sin(a)
            world.backend.initialize_pose(rx, ry, math.atan2(fy - ry, fx - rx))
            for _ in range(15):
                world.step()
            pose = evaluator.true_pose(world)
            frame = world.capture("probe")
            ds = [d for d in det.detect(frame.rgb, pose=tuple(pose)) if d.position is not None and not d.partial]
            if not ds:
                continue
            best = min(ds, key=lambda d: math.dist(d.position, (fx, fy)))
            size = min(best.width_m or 0.2, 1.1 * (best.height_m or 0.2))
            radius = float(np.clip(size / 2, 0.05, 0.15))
            gap = forward_gap(pose, item)
            old = nudge_creep(None, best.range_m, radius)
            new = nudge_creep(best.contact_range_m, best.range_m, radius)
            rows.append((name, rng, ang, round(gap, 3), round(old, 3), round(new, 3), round(best.fill, 2)))
print("fixture range angle true_gap old_creep new_creep fill")
for r in rows:
    print(*r)
for name in world.fixtures:
    sel = [r for r in rows if r[0] == name]
    if not sel:
        continue
    reach_old = sum(r[4] >= r[3] for r in sel)
    reach_new = sum(r[5] >= r[3] for r in sel)
    over_new = [r[5] - r[3] for r in sel]
    print(f"SUMMARY {name}: views {len(sel)}  reaches old {reach_old}/{len(sel)}  new {reach_new}/{len(sel)}  "
          f"new depth past surface mean {np.mean(over_new):.3f} max {np.max(over_new):.3f} "
          f"(push depth {PUSH_DEPTH})")
