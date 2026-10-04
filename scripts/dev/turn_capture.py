"""Capture in-place turns for the offline turn-drift benchmark (evaluation only).

The robot is put at sampled free positions (explicit pose reset, as evaluation
scripts may) stratified by the distance to the nearest non-wall object, and turns
in place under scripted commands. Per 0.1 s control period the onboard image, the
command that was in force during the preceding period (exactly what the runtime
passes to the VSLAM) and the true pose at the image instant are stored.

Modes:
  driven  commands go through the wheel backend; slip, base wander and chassis
          motion come from physics (primary)
  reset   the pose is set every frame to theta0 + slip * integrated command, so the
          slip ratio is controlled (ablation: --slip)

Patterns: full2 (>= 720 deg true), survey (360 deg, 0.4 s stops every 45 deg), sweep
(+50/-100/+50 deg), back (+180/-180 deg). --approach drives straight first.

    PYTHONPATH=src python scripts/dev/turn_capture.py work/evidence/turncap-arena-driven --world arena \\
        --map-dir work/evidence/i3-nocov-arena-s2/map --patterns full2 survey sweep back --per-bin 6
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
from learning import map_origin_for  # noqa: E402

from amr_rl.control.contract import DriveCommand  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.world import SimWorld, load_world_config  # noqa: E402

PERIOD = 0.1


def pattern_commands(name, w):
    """[(v, w, seconds)] after 0.5 s standing still."""
    still = [(0.0, 0.0, 0.5)]
    t45 = math.radians(45) / w
    if name == "full2":  # >= 720 deg of true rotation down to slip 0.75
        return still + [(0.0, w, 4 * math.pi / (0.75 * w))]
    if name == "survey":
        return still + [seg for _ in range(8) for seg in ((0.0, w, t45), (0.0, 0.0, 0.4))]
    if name == "sweep":
        a = math.radians(50) / w
        return still + [(0.0, w, a), (0.0, -w, 2 * a), (0.0, w, a)]
    if name == "back":
        return still + [(0.0, w, math.pi / w), (0.0, -w, math.pi / w)]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--world", default="arena")
    ap.add_argument("--map-dir", default=None, help="saved map for the map-frame truth (onmap replay)")
    ap.add_argument("--origin", nargs=3, type=float, default=None,
                    help="world pose of the map frame when the map's result.json has none (navigation maps: the start)")
    ap.add_argument("--mode", default="driven", choices=["driven", "reset"])
    ap.add_argument("--rate", type=float, default=0.45, help="commanded turn rate (rad/s), as survey/coverage turns")
    ap.add_argument("--slip", type=float, default=0.83, help="reset mode: true rotation / commanded")
    ap.add_argument("--patterns", nargs="+", default=["full2", "survey", "sweep", "back"])
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--bins", nargs="+", type=float, default=[0.0, 0.4, 0.7, 9.0])
    ap.add_argument("--clearance", type=float, default=0.28)
    ap.add_argument("--max-kf-dist", type=float, default=0.3,
                    help="with --map-dir: only positions within this of a map keyframe (where the robot has been)")
    ap.add_argument("--approach", type=float, default=0.0, help="m driven straight at 0.15 m/s before turning")
    ap.add_argument("--depth", action="store_true", help="also store rendered depth (mm, uint16) for scoring")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = SimWorld(load_world_config(args.world), inspection=False, seed=args.seed)
    for _ in range(20):
        world.step()
    steps = int(round(PERIOD / world.dt))
    origin = None
    if args.map_dir:
        origin = np.asarray(args.origin if args.origin else map_origin_for(Path(args.map_dir)), float)
    lx, ly = world.config["room"]["size"]
    rng = np.random.default_rng(args.seed)
    grid = [(float(x), float(y)) for x in np.arange(-lx / 2, lx / 2, 0.05) for y in np.arange(-ly / 2, ly / 2, 0.05)]
    free = [p for p in grid if not capture_common.blocked(world, p, args.clearance)]
    if origin is not None:
        kfs = json.loads((Path(args.map_dir) / "map.json").read_text())["keyframes"]
        kf_world = np.array([evaluator.map_to_world(origin, k["pose"][:2]) for k in kfs])
        free = [p for p in free if capture_common.keyframe_distance(kf_world, p) <= args.max_kf_dist]
    dist = np.array([capture_common.nearest_object(world, p) for p in free])
    positions = []
    for lo, hi in zip(args.bins[:-1], args.bins[1:]):
        idx = np.flatnonzero((dist >= lo) & (dist < hi))
        for i in rng.permutation(idx)[:args.per_bin]:
            positions.append({"xy": list(free[i]), "object_m": float(dist[i]), "bin": f"{lo}-{hi}",
                              "heading0": float(rng.uniform(-math.pi, math.pi))})
    meta = {"world": args.world, "map_dir": args.map_dir, "origin_world": None if origin is None else origin.tolist(),
            "mode": args.mode, "rate": args.rate, "slip": args.slip if args.mode == "reset" else None,
            "approach": args.approach, "period": PERIOD, "seed": args.seed, "sequences": []}
    gen = 0
    for n, pos in enumerate(positions):
        for pattern in args.patterns:
            seq_id = f"p{n:02d}-{pattern}"
            folder = out / seq_id
            folder.mkdir(exist_ok=True)
            x0, y0, th0 = pos["xy"][0], pos["xy"][1], pos["heading0"]
            plan = pattern_commands(pattern, args.rate)
            if args.approach > 0:  # back off along -heading so the approach ends at the position
                x0, y0 = x0 - args.approach * math.cos(th0), y0 - args.approach * math.sin(th0)
                if capture_common.blocked(world, (x0, y0), args.clearance):
                    x0, y0 = pos["xy"]
                    approach = []
                else:
                    approach = [(0.15, 0.0, args.approach / 0.15)]
                plan = [(0.0, 0.0, 0.5)] + approach + plan
            world.backend.initialize_pose(x0, y0, th0)
            for _ in range(30):
                world.step()
            frames, cmd_prev, theta_cmd = [], (0.0, 0.0), th0
            schedule = [(v, w) for v, w, sec in plan for _ in range(int(round(sec / PERIOD)))]
            for k, (v, w) in enumerate(schedule + [(0.0, 0.0)]):
                if args.mode == "reset":
                    theta_cmd += args.slip * cmd_prev[1] * PERIOD
                    world.backend.initialize_pose(x0, y0, theta_cmd)
                    world.step()
                world.camera.move_to_attach()
                if args.depth:
                    rgb, depth, _, _ = world.camera.render(rgb=True, depth=True)
                else:
                    rgb = world.camera.render(rgb=True)[0]
                truth = evaluator.true_pose(world)
                rec = {"file": f"{seq_id}/{k:04d}.png", "t": round(k * PERIOD, 3), "cmd": [float(cmd_prev[0]), float(cmd_prev[1])],
                       "world": [float(v_) for v_ in truth],
                       "object_m": capture_common.nearest_object(world, truth[:2], heading=float(truth[2]))}
                if origin is not None:
                    rec["map"] = [float(v_) for v_ in evaluator.to_map_frame(origin, truth)]
                cv2.imwrite(str(out / rec["file"]), np.asarray(rgb)[..., ::-1])
                if args.depth:
                    mm = np.clip(np.asarray(depth) * 1000.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(folder / f"{k:04d}_depth.png"), mm)
                    rec["depth"] = f"{seq_id}/{k:04d}_depth.png"
                frames.append(rec)
                if k == len(schedule):
                    break
                if args.mode == "driven":
                    now = world.time
                    gen += 1
                    world.backend.command(DriveCommand(v, w, now, now + 0.2, gen, "manual"), now=now)
                    for _ in range(steps):
                        world.step()
                cmd_prev = (v, w)
            world.backend.stop()
            meta["sequences"].append({"id": seq_id, "position": n, "pattern": pattern, **pos, "frames": frames})
            true_rot = sum(abs(math.atan2(math.sin(b["world"][2] - a["world"][2]), math.cos(b["world"][2] - a["world"][2])))
                           for a, b in zip(frames[:-1], frames[1:]))
            cmd_rot = sum(abs(f["cmd"][1]) * PERIOD for f in frames)
            print(f"{seq_id} bin {pos['bin']} object {pos['object_m']:.2f} m frames {len(frames)} "
                  f"turned {math.degrees(true_rot):.0f} deg (commanded {math.degrees(cmd_rot):.0f})", flush=True)
    (out / "sequences.json").write_text(json.dumps(meta))
    print("done", len(meta["sequences"]))


if __name__ == "__main__":
    main()
