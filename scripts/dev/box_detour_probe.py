"""Replay of the round-5 box contact (evaluation only): a saved map, a start pose, a
box put down near the route, an operator goal past it. Reports the true minimum
footprint-to-box gap, contacts and the goal result for several lateral box offsets.

Default scenario: navlc-20261003-on-home_a seed 1, the "obstacle placed on mapped
route" test (contact at t=425.7 s).

    PYTHONPATH=src python scripts/dev/box_detour_probe.py out.jsonl --lateral -0.2 -0.1 0 0.1 0.2
    ... --shadow-depth 0      # A/B: the guard without shadowing hidden faces
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from navigation import enable, run_goal, wait_tracking  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.navigation.near_field import GuardConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--world", default="home_a")
ap.add_argument("--map", default="work/evidence/navlc-20261003-on-home_a/home_a-s1/map")
ap.add_argument("--origin", nargs=3, type=float, default=[-1.6, -0.6, 1.5566370614359173])
ap.add_argument("--start-map", nargs=3, type=float, default=[1.377, -0.194, -0.43])
ap.add_argument("--goal-map", nargs=2, type=float, default=[1.075, -1.475])
ap.add_argument("--box-world", nargs=2, type=float, default=[-0.6704589164649349, 0.9620221370122102])
ap.add_argument("--lateral", nargs="+", type=float, default=[0.0])
ap.add_argument("--shadow-depth", type=float, default=None)
ap.add_argument("--no-loop-closure", action="store_true")
args = ap.parse_args()

origin = np.array(args.origin)
start_xy = evaluator.map_to_world(origin, args.start_map[:2])
start = [float(start_xy[0]), float(start_xy[1]), float(origin[2] + args.start_map[2])]
goal_world = evaluator.map_to_world(origin, args.goal_map)
route = np.asarray(goal_world) - np.asarray(start[:2])
normal = np.array([-route[1], route[0]]) / max(np.linalg.norm(route), 1e-9)

for lateral in args.lateral:
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="operator_only",
                        place_descriptor=None if args.no_loop_closure else "megaloc")
    run = Path(args.out).with_suffix("") / f"lat{lateral:+.2f}"
    s = Session(args.world, run_dir=run, memory_path=run / "m.sqlite", config=cfg, inspection=False,
                map_dir=args.map, map_origin=origin.tolist(), world_overrides={"robot_start": start})
    if args.shadow_depth is not None and s.runtime.guard is not None:
        s.runtime.guard.cfg = GuardConfig(**{**s.runtime.guard.cfg.__dict__, "shadow_depth": args.shadow_depth})
    log = []
    ok = wait_tracking(s, 40.0, log)
    rec = {"lateral": lateral, "relocalised": ok}
    if ok:
        enable(s, log, "probe start")
        box = np.asarray(args.box_world) + lateral * normal
        s.world.place_evaluation_obstacle(box)
        for _ in range(3):
            s.control_step()
        res = run_goal(s, {"name": "past_box", "expect": "reach", "map_xy": list(args.goal_map), "xy": None}, log,
                       obstacle=(box, s.world.evaluation_obstacle_size))
        rec.update({k: res.get(k) for k in ("result", "navigation_status", "navigation_reason", "min_obstacle_gap_m",
                                            "min_obstacle_gap_t", "true_final_distance", "guard_during_goal")})
        rec["contact"] = any("evaluation_obstacle" in c["with"] for c in res.get("contacts", []))
        rec["recovery"] = res.get("recovery_seen", False)
    rec["shadow_depth"] = None if s.runtime.guard is None else s.runtime.guard.cfg.shadow_depth
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    s.close()
