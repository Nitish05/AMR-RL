"""Capture onboard frames with ground-truth object boxes for detector benchmarks
(evaluation only; perception swap, steps 5a/5b).

The robot is placed (explicit pose reset, as evaluation scripts may) at sampled
poses facing a fixture at 0.4-2.0 m, every fixture's flag is set at random
(none / yellow / red, outside the response rules), and the onboard camera renders
RGB plus Genesis' link-level segmentation. Ground truth (scoring only) per frame:
each fixture's visible body box, its true flag state and whether the raised flag
is visible, the fixture's true position and the robot's true pose.
``scripts/eval/detector_bench.py`` scores detectors on the saved frames.

    PYTHONPATH=src python scripts/dev/detector_capture.py work/evidence/detcap-arena_textured --world arena_textured --frames 300
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from amr_rl.sim import evaluator
from amr_rl.sim.world import SimWorld, load_world_config

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--world", default="arena_textured")
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-range", type=float, default=0.4)
ap.add_argument("--max-range", type=float, default=2.0)
ap.add_argument("--clearance", type=float, default=0.3)
ap.add_argument("--min-pixels", type=int, default=150, help="smaller visible bodies are not ground-truth objects")
args = ap.parse_args()

out = Path(args.out)
(out / "frames").mkdir(parents=True, exist_ok=True)
world = SimWorld(load_world_config(args.world), inspection=False, seed=args.seed)
for _ in range(20):
    world.step()
lx, ly = world.config["room"]["size"]
rng = np.random.default_rng(args.seed)

# segmentation index -> (fixture name, part)
seg_parts = {}
for idx, key in world.scene.segmentation_idx_dict.items():
    if not isinstance(key, tuple):
        continue
    for name, (item, entity) in world.fixtures.items():
        if key[0] != entity.idx:
            continue
        part = "body"
        if item["shape"] != "sphere":
            for color in ("yellow", "red"):
                if key[1] == entity.get_link(f"{color}_panel").idx:
                    part = color
        seg_parts[idx] = (name, part)


def blocked(p, r):
    """Scoring geometry: does a disc at p overlap walls, furniture or fixtures?"""
    if abs(p[0]) > lx / 2 - r or abs(p[1]) > ly / 2 - r:
        return True
    boxes = [g for g in world.static_geometry if g["name"] != "evaluation_obstacle"]
    for item, entity in world.fixtures.values():
        boxes.append({"size": item["size"], "xy": np.asarray(entity.get_pos()).reshape(3)[:2],
                      "yaw": float(item.get("yaw", 0.0)), "round": item["shape"] in ("cylinder", "sphere")})
    for b in boxes:
        d = np.asarray(p, float) - np.asarray(b["xy"], float)
        if b.get("round"):
            if np.linalg.norm(d) <= b["size"][0] / 2 + r:
                return True
            continue
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        loc = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        if np.linalg.norm(np.maximum(np.abs(loc) - np.asarray(b["size"][:2]) / 2, 0.0)) <= r:
            return True
    return False


names = list(world.fixtures)
flagged = [n for n in names if world.fixtures[n][0]["shape"] != "sphere"]
frames = []
while len(frames) < args.frames:
    target = names[int(rng.integers(len(names)))]
    fxy = np.asarray(world.fixtures[target][1].get_pos()).reshape(3)[:2]
    rng_m, around = rng.uniform(args.min_range, args.max_range), rng.uniform(-math.pi, math.pi)
    p = fxy + rng_m * np.array([math.cos(around), math.sin(around)])
    if blocked(p, args.clearance):
        continue
    heading = math.atan2(fxy[1] - p[1], fxy[0] - p[0]) + math.radians(rng.uniform(-25, 25))
    flags = {n: [None, "yellow", "red"][int(rng.choice(3, p=[0.4, 0.3, 0.3]))] for n in flagged}
    for n, c in flags.items():
        world.raise_flag_for_evaluation(n, c)
    world.backend.initialize_pose(float(p[0]), float(p[1]), heading)
    for _ in range(40):  # panels travel; robot settles
        world.step()
    world.camera.move_to_attach()
    rgb, _, seg, _ = world.camera.render(rgb=True, segmentation=True)
    seg = np.asarray(seg)
    H, W = seg.shape[:2]
    objects = []
    for name in names:
        masks = {part: np.zeros((H, W), bool) for part in ("body", "yellow", "red")}
        for idx, (n, part) in seg_parts.items():
            if n == name:
                masks[part] |= seg == idx
        if masks["body"].sum() < args.min_pixels:
            continue
        ys, xs = np.nonzero(masks["body"])
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        state = flags.get(name)
        flag_px = int(masks[state].sum()) if state else 0
        objects.append({
            "name": name, "box": box, "pixels": int(masks["body"].sum()),
            "truncated": bool(box[0] <= 1 or box[1] <= 1 or box[2] >= W - 1 or box[3] >= H - 1),
            "flag": state, "flag_pixels": flag_px,
            "xy_world": [float(v) for v in np.asarray(world.fixtures[name][1].get_pos()).reshape(3)[:2]],
        })
    k = len(frames)
    cv2.imwrite(str(out / "frames" / f"{k:04d}.png"), np.asarray(rgb)[..., ::-1])
    frames.append({"file": f"frames/{k:04d}.png", "target": target, "pose_world": [float(v) for v in evaluator.true_pose(world)],
                   "objects": objects})
    if k % 25 == 0:
        print(f"frame {k} target {target} objects {[o['name'] for o in objects]}", flush=True)

(out / "frames.json").write_text(json.dumps({"world": args.world, "seed": args.seed, "frames": frames}, indent=1))
print("done", len(frames))
