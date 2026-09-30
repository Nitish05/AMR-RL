"""Detector localisation error for fixtures seen from known poses (evaluation probe:
uses the true robot pose so only the detector's geometry is measured)."""

import math
import sys

import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.entities import FixtureDetector
from amr_rl.sim import evaluator
from amr_rl.sim.world import SimWorld, load_world_config

world = SimWorld(load_world_config(sys.argv[1] if len(sys.argv) > 1 else "arena"), inspection=False)
det = FixtureDetector(CameraModel.from_spec(world.spec))
rows = []
for name, (item, _) in world.fixtures.items():
    fx, fy = item["pos"]
    for rng in (0.7, 1.2, 1.8):
        for ang in (0, 35, 70, 110, 160, 220, 290):
            a = math.radians(ang)
            rx, ry = fx + rng * math.cos(a), fy + rng * math.sin(a)
            world.backend.initialize_pose(rx, ry, math.atan2(fy - ry, fx - rx))
            for _ in range(15):
                world.step()
            pose = evaluator.true_pose(world)
            frame = world.capture("probe")
            ds = [d for d in det.detect(frame.rgb, pose=tuple(pose)) if d.position is not None]
            if not ds:
                rows.append((name, rng, ang, None, None, None))
                continue
            best = min(ds, key=lambda d: math.dist(d.position, (fx, fy)))
            rows.append((name, rng, ang, round(math.dist(best.position, (fx, fy)), 3), round(best.fill, 2),
                         round(best.width_m or 0, 2)))
for r in rows:
    print(*r)
for name in world.fixtures:
    e = [r[3] for r in rows if r[0] == name and r[3] is not None]
    print("SUMMARY", name, "n", len(e), "mean", round(float(np.mean(e)), 3), "max", round(float(np.max(e)), 3))
