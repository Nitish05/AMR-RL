"""Capture onboard frames for the offline relocalisation probe (evaluation only).

Places the robot (explicit pose reset, as evaluation scripts may) at the arena
learning start poses and at sampled free positions stratified by distance to the
nearest keyframe of a saved map, and renders a full in-place turn in fixed heading
steps at each, the way the operator turn at a phase start sees the room. Ground
truth (the world pose and the same pose in the map frame) is stored for scoring
only; ``reloc_probe.py`` replays the frames through the VSLAM.

    PYTHONPATH=src python scripts/dev/reloc_capture.py work/evidence/<map-…>/map work/evidence/reloc-capture-<stamp>
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_common  # noqa: E402
from learning import ARENA_STARTS, map_origin_for  # noqa: E402

from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.world import SimWorld, load_world_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("map_dir")
ap.add_argument("out")
ap.add_argument("--world", default="arena")
ap.add_argument("--per-bin", type=int, default=8)
ap.add_argument("--step-deg", type=float, default=4.0)
ap.add_argument("--clearance", type=float, default=0.32, help="free radius around a sampled position (m)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-starts", action="store_true", help="skip the arena learning start poses")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
map_dir = Path(args.map_dir)
origin = np.asarray(map_origin_for(map_dir), float)
keyframes = np.array([k["pose"] for k in json.loads((map_dir / "map.json").read_text())["keyframes"]], float)
kf_world = np.array([evaluator.map_to_world(origin, k[:2]) for k in keyframes])

world = SimWorld(load_world_config(args.world), inspection=False)
for _ in range(20):
    world.step()
lx, ly = world.config["room"]["size"]


def blocked(p, r):
    return capture_common.blocked(world, p, r)


def kf_distance(p):
    return capture_common.keyframe_distance(kf_world, p)


positions = []
if not args.no_starts and args.world == "arena":
    for k, s in enumerate(ARENA_STARTS):
        positions.append({"kind": "start", "start_index": k, "xy": [s[0], s[1]], "heading0": s[2]})
rng = np.random.default_rng(args.seed)
grid = [(x, y) for x in np.arange(-lx / 2, lx / 2, 0.1) for y in np.arange(-ly / 2, ly / 2, 0.1)]
free = [p for p in grid if not blocked(p, args.clearance)]
bins = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 9.0)]
for lo, hi in bins:
    cand = [p for p in free if lo <= kf_distance(p) < hi]
    for i in rng.permutation(len(cand))[:args.per_bin]:
        positions.append({"kind": "sample", "xy": [float(cand[i][0]), float(cand[i][1])],
                          "heading0": float(rng.uniform(-math.pi, math.pi))})

steps = int(round(360 / args.step_deg))
for n, pos in enumerate(positions):
    pos["id"] = n
    pos["kf_distance"] = kf_distance(pos["xy"])
    pos["frames"] = []
    folder = out / f"p{n:03d}"
    folder.mkdir(exist_ok=True)
    for k in range(steps):
        th = pos["heading0"] + math.radians(args.step_deg) * k
        world.backend.initialize_pose(pos["xy"][0], pos["xy"][1], th)
        for _ in range(5):  # settle
            world.step()
        frame = world.capture("probe")
        truth = evaluator.true_pose(world)
        cv2.imwrite(str(folder / f"{k:03d}.png"), frame.rgb[..., ::-1])
        pos["frames"].append({"file": f"p{n:03d}/{k:03d}.png", "world": [float(v) for v in truth],
                              "map": [float(v) for v in evaluator.to_map_frame(origin, truth)]})
    print(f"position {n} {pos['kind']} xy={np.round(pos['xy'], 2).tolist()} kf_dist={pos['kf_distance']:.2f}", flush=True)

(out / "poses.json").write_text(json.dumps({
    "map_dir": str(map_dir), "world": args.world, "origin_world": origin.tolist(), "step_deg": args.step_deg,
    "clearance": args.clearance, "bins": bins, "positions": positions}, indent=1))
print(f"wrote {len(positions)} positions x {steps} frames to {out}")
