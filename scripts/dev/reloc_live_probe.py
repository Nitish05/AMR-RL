"""Live relocalisation probe (evaluation only): start the robot on a saved map at
sampled free positions, binned by the distance to the nearest map keyframe, and
measure whether it relocalises and whether the accepted pose is right.

Per position: a fresh session on the map (memory empty, autonomy off). The operator
turns the robot slowly in place, as the learning restarts do, until the VSLAM tracks
or ``--timeout`` passes. Scored against simulator truth:
- good: within 0.10 m and 5 deg;
- false: beyond 0.2 m or 8.6 deg;
- imprecise: in between.
Also recorded: the degrees turned before acceptance.

Positions are chosen from the world config before any run. They are cells of a
0.25 m lattice at least ``--clearance`` from walls, obstacles and fixtures, with
seeded headings, up to ``--per-bin`` per distance bin (< 0.3, 0.3-0.5, >= 0.5 m).

    PYTHONPATH=src python scripts/dev/reloc_live_probe.py work/evidence/<run>/map out_dir \\
        --world arena --odometry imu_encoders
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import add_runtime_args, apply_runtime_args, provenance  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402
from amr_rl.sim.world import load_world_config  # noqa: E402

BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 9.0))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def free_positions(cfg, clearance, step=0.25):
    lx, ly = cfg["room"]["size"]
    boxes = [(o["pos"], o["size"], float(o.get("yaw", 0.0))) for o in cfg.get("obstacles", [])]
    boxes += [(f["pos"], f["size"], float(f.get("yaw", 0.0))) for f in cfg.get("fixtures", [])]
    out = []
    for x in np.arange(-lx / 2 + clearance, lx / 2 - clearance + 1e-9, step):
        for y in np.arange(-ly / 2 + clearance, ly / 2 - clearance + 1e-9, step):
            ok = True
            for (bx, by), size, yaw in boxes:
                c, s = math.cos(yaw), math.sin(yaw)
                dx, dy = x - bx, y - by
                u, v = c * dx + s * dy, -s * dx + c * dy
                ex = max(abs(u) - size[0] / 2, 0.0)
                ey = max(abs(v) - size[1] / 2, 0.0)
                if math.hypot(ex, ey) < clearance:
                    ok = False
                    break
            if ok:
                out.append((float(x), float(y)))
    return out


def keyframes_world(map_dir, origin):
    kfs = json.loads((Path(map_dir) / "map.json").read_text())["keyframes"]
    T = evaluator.planar_T(*origin)
    pts = np.array([k["pose"][:2] for k in kfs], float)
    return pts @ T[:2, :2].T + T[:2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("out")
    ap.add_argument("--world", default="arena")
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--clearance", type=float, default=0.40)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    add_runtime_args(ap)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    origin = json.loads((Path(args.map).parent / "result.json").read_text())["origin_world"]
    cfg = load_world_config(args.world)
    kf = keyframes_world(args.map, origin)
    rng = np.random.default_rng(args.seed)
    cands = free_positions(cfg, args.clearance)
    rng.shuffle(cands)
    chosen = []
    for lo, hi in BINS:
        sel = [p for p in cands if lo <= float(np.min(np.hypot(*(kf - np.array(p)).T))) < hi][:args.per_bin]
        chosen += [(p, lo, hi) for p in sel]
    plan = [{"xy": p, "heading": float(rng.uniform(-math.pi, math.pi)), "bin": [lo, hi],
             "kf_distance": float(np.min(np.hypot(*(kf - np.array(p)).T)))} for p, lo, hi in chosen]
    provenance(out, extra={"args": vars(args), "map_origin_world": origin, "plan": plan})
    results = []
    for i, item in enumerate(plan):
        rc = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="learned", seed=args.seed,
                           initial_survey=False)
        apply_runtime_args(rc, args)
        start = [item["xy"][0], item["xy"][1], item["heading"]]
        s = Session(args.world, run_dir=out / f"p{i}" / "session", memory_path=out / f"p{i}" / "mem.sqlite",
                    config=rc, seed=args.seed + i, inspection=False, map_dir=args.map, map_origin=origin,
                    world_overrides={"robot_start": start})
        rt = s.runtime
        turned, accepted = 0.0, None
        prev = evaluator.true_pose(s.world)
        while s.now < args.timeout:
            rt.command({"action": "manual", "v": 0.0, "w": 0.4, "generation": rt.supervisor.generation})
            s.control_step()
            now_pose = evaluator.true_pose(s.world)
            turned += abs(wrap(now_pose[2] - prev[2]))
            prev = now_pose
            if rt.slam.status == "tracking":
                row = s.truth[-1]
                accepted = {"t": s.now, "turned_deg": math.degrees(turned), "err_m": row["err"],
                            "herr_deg": None if row["herr"] is None else math.degrees(abs(row["herr"]))}
                break
        if accepted is None:
            verdict = "none"
        elif accepted["err_m"] <= 0.10 and accepted["herr_deg"] <= 5.0:
            verdict = "good"
        elif accepted["err_m"] > 0.2 or accepted["herr_deg"] > math.degrees(0.15):
            verdict = "false"
        else:
            verdict = "imprecise"
        results.append({**item, "verdict": verdict, "accepted": accepted})
        print(json.dumps(results[-1]), flush=True)
        s.close()
    summary = {f"{lo}-{hi}": {v: sum(1 for r in results if r["bin"] == [lo, hi] and r["verdict"] == v)
                              for v in ("good", "imprecise", "false", "none")} for lo, hi in BINS}
    (out / "result.json").write_text(json.dumps({"summary": summary, "positions": results}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
