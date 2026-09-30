"""Render documentation images: a close third-person view of Pip beside a fixture
(inspection camera only; never used by the robot) and the matching onboard frame.

    PYTHONPATH=src python scripts/dev/media.py --out docs/media
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from amr_rl.expression.policy import expression_from
from amr_rl.expression.screen import render
from amr_rl.sim.world import SimWorld, load_world_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    world = SimWorld(load_world_config("dev_interact"), inspection=True)
    item, _ = world.fixtures["bloom"]
    fx, fy = item["pos"]
    rx, ry = fx - 0.8, fy - 0.15
    world.backend.initialize_pose(rx, ry, math.atan2(fy - ry, fx - rx))
    for _ in range(40):
        world.step()
    face = render(expression_from({"activity": {"name": "engage"},
                                   "interaction": {"action": "signal", "status": "acting"}}))
    world.set_screen(face, signal_pattern=False)
    # Close three-quarter view from in front of the robot, looking back at it.
    world.inspection.set_pose(pos=(rx + 0.6, ry - 0.7, 0.5), lookat=(rx + 0.1, ry, 0.14), up=(0.0, 0.0, 1.0))
    img = world.render_inspection()
    cv2.imwrite(str(out / "robot-inspection.png"), img[..., ::-1])
    onboard = world.capture("media").rgb
    cv2.imwrite(str(out / "onboard-frame.png"), onboard[..., ::-1])
    print("wrote", out / "robot-inspection.png", out / "onboard-frame.png", np.asarray(img).shape)


if __name__ == "__main__":
    main()
