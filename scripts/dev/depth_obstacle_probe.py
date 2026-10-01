"""Offline accuracy of the monocular-depth near-field detector (evaluation probe).

Renders the onboard camera from sampled true poses in a room, with and without the
evaluation box placed straight ahead, and compares the nearest obstacle the detector
reports in the corridor ahead (x 0.15-1.0 m, |y| <= 0.2 m in the robot frame) with
the true nearest obstacle there (static geometry, fixtures, walls). Ground truth is
used for placing the robot/box and scoring only.

    PYTHONPATH=src python scripts/dev/depth_obstacle_probe.py home_a out.json --poses 40
"""

import argparse
import json
import math
import time

import numpy as np

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.near_depth import MonoDepthObstacles, load_backend
from amr_rl.sim import evaluator
from amr_rl.sim.world import SimWorld, load_world_config

ap = argparse.ArgumentParser()
ap.add_argument("world")
ap.add_argument("out")
ap.add_argument("--poses", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--box-dists", nargs="+", type=float, default=[0.45, 0.7])
args = ap.parse_args()

world = SimWorld(load_world_config(args.world), inspection=False)
cam = CameraModel.from_spec(world.spec)
backend = load_backend()
assert backend is not None, "model not in the local cache: run scripts/amr.sh fetch-depth-model"
det = MonoDepthObstacles(cam, backend=backend)
rng = np.random.default_rng(args.seed)
lx, ly = world.config["room"]["size"]
box_size = world.evaluation_obstacle_size
X, Y = np.meshgrid(np.arange(0.15, 1.0, 0.02), np.arange(-0.2, 0.201, 0.02))
corridor = np.stack([X.ravel(), Y.ravel()], 1)


def inside_geometry(pts, extra=()):
    occ = np.zeros(len(pts), bool)
    boxes = [g for g in world.static_geometry if g["name"] != "evaluation_obstacle"] + list(extra)
    for item, entity in world.fixtures.values():
        xy = np.asarray(entity.get_pos()).reshape(3)[:2]
        boxes.append({"size": item["size"], "xy": xy, "yaw": float(item.get("yaw", 0.0)),
                      "round": item["shape"] in ("cylinder", "sphere")})
    for b in boxes:
        d = pts - np.asarray(b["xy"])[None]
        if b.get("round"):
            occ |= np.linalg.norm(d, axis=1) <= b["size"][0] / 2
            continue
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        loc = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
        occ |= (np.abs(loc[:, 0]) <= b["size"][0] / 2) & (np.abs(loc[:, 1]) <= b["size"][1] / 2)
    occ |= (np.abs(pts[:, 0]) > lx / 2) | (np.abs(pts[:, 1]) > ly / 2)
    return occ


def true_nearest(pose, extra=()):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    w = np.stack([pose[0] + c * corridor[:, 0] - s * corridor[:, 1],
                  pose[1] + s * corridor[:, 0] + c * corridor[:, 1]], 1)
    occ = inside_geometry(w, extra)
    return float(corridor[occ, 0].min()) if occ.any() else None


def detected_nearest(points, min_points=6):
    if points is None or len(points) == 0:
        return None
    m = (points[:, 0] >= 0.15) & (points[:, 0] <= 1.0) & (np.abs(points[:, 1]) <= 0.2)
    p = points[m]
    if len(p) < min_points:
        return None
    xs = np.sort(p[:, 0])
    return float(xs[min_points - 1])  # robust nearest: the min_points-th closest


def robot_fits(x, y):
    probe = np.array([[x + dx, y + dy] for dx in (-0.25, 0, 0.25) for dy in (-0.25, 0, 0.25)])
    return not inside_geometry(probe).any()


def render(pose):
    world.backend.initialize_pose(*pose)
    for _ in range(12):
        world.step()
    tp = evaluator.true_pose(world)
    return world.capture("probe").rgb, tp


rows = []
park = None
t0 = time.time()
while len([r for r in rows if r["box"] is None]) < args.poses:
    x, y = rng.uniform(-lx / 2 + 0.4, lx / 2 - 0.4), rng.uniform(-ly / 2 + 0.4, ly / 2 - 0.4)
    th = rng.uniform(-math.pi, math.pi)
    if not robot_fits(x, y):
        continue
    for dist in [None] + list(args.box_dists):
        extra = ()
        if dist is not None:
            bxy = (x + (dist + box_size[0] / 2) * math.cos(th), y + (dist + box_size[0] / 2) * math.sin(th))
            if inside_geometry(np.array([bxy]))[0]:
                continue
            world.evaluation_obstacle.set_pos([bxy[0], bxy[1], box_size[2] / 2 + 0.002], zero_velocity=True)
            extra = ({"size": box_size, "xy": bxy, "yaw": 0.0},)
        else:
            world.evaluation_obstacle.set_pos([lx + 3.0, 0.0, box_size[2] / 2], zero_velocity=True)
        rgb, tp = render((x, y, th))
        out = det.detect(rgb)
        truth = true_nearest(tp, extra)
        rows.append({"pose": [float(v) for v in tp], "box": dist, "true_nearest": truth,
                     "detected_nearest": detected_nearest(out["points"]),
                     "n_points": int(len(out["points"])), "fit": out["fit"]})
print(f"frames {len(rows)} in {time.time() - t0:.0f} s")
json.dump(rows, open(args.out, "w"), indent=1)


def summary(sel, label):
    near = [r for r in sel if r["true_nearest"] is not None and r["true_nearest"] <= 0.6]
    clear = [r for r in sel if r["true_nearest"] is None or r["true_nearest"] > 1.0]
    hit = [r for r in near if r["detected_nearest"] is not None and r["detected_nearest"] <= 0.8]
    fa = [r for r in clear if r["detected_nearest"] is not None and r["detected_nearest"] <= 0.6]
    err = [r["detected_nearest"] - r["true_nearest"] for r in hit]
    print(f"{label}: obstacle within 0.6 m: detected (<=0.8 m) {len(hit)}/{len(near)}"
          + (f", range error median {np.median(err):+.3f} m, max |err| {np.max(np.abs(err)):.3f}" if err else "")
          + f" | corridor clear to 1.0 m: false alarms (<=0.6 m) {len(fa)}/{len(clear)}")


summary(rows, "all frames")
summary([r for r in rows if r["box"] is not None], "box frames")
summary([r for r in rows if r["box"] is None], "no-box frames")
