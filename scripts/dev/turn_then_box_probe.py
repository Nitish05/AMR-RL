"""Reproducible obstacle-on-route scenarios (evaluation probe).

The failing geometry in round 2: the robot first turns in place toward its route
(no translation, so no parallax evidence), then drives, and the box sits about 0.6 m
ahead (as close as the navigation evaluation places it). Each scenario:

1. load a saved navigation map, relocalise at that run's start;
2. drive to a certified point P (operator goal);
3. pick a direction ``turn`` degrees from the robot's heading, a goal 1.5 m away
   along it and a box ``dist`` metres along it (both inside certified free space);
4. place the box, send the goal, and record contacts, holds and the outcome.

Ground truth is used to place the box and to score contacts only.

    PYTHONPATH=src python scripts/dev/turn_then_box_probe.py <nav run dir> <out json> \
        --turns 180 135 90 --dists 0.6 0.75
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import navigation as nav_eval  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("out")
ap.add_argument("--turns", nargs="+", type=float, default=[180, 135, 90])
ap.add_argument("--dists", nargs="+", type=float, default=[0.6, 0.75])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--guard", choices=["on", "off", "observe"], default="on",
                help="near-field depth guard: active, absent, or detecting without acting")
ap.add_argument("--trace", action="store_true", help="log every control step after the box is placed")
args = ap.parse_args()
run_dir = Path(args.run_dir)
res = json.loads((run_dir / "result.json").read_text())
start = res["start_world"]
results = []
for turn in args.turns:
    for dist in args.dists:
        work = Path(args.out).with_suffix("") / f"t{int(turn)}-d{int(dist * 100)}"
        work.mkdir(parents=True, exist_ok=True)
        cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only",
                            seed=args.seed)
        s = Session(res["world"], run_dir=str(work), memory_path=str(work / "m.sqlite"), config=cfg,
                    seed=args.seed, inspection=False, map_dir=str(run_dir / "map"), map_origin=start,
                    world_overrides={"robot_start": start})
        rt = s.runtime
        if args.guard == "off":
            rt.guard = rt.nav.guard = None
        elif args.guard == "observe" and rt.guard is not None:
            rt.guard.cfg.observe_only = True
        log = []
        row = {"turn": turn, "dist": dist}
        if not nav_eval.wait_tracking(s, 40, log):
            row["result"] = "skipped: never relocalised"
            results.append(row)
            print(row, flush=True)
            s.close()
            continue
        rt.chooser.policy = "operator_only"
        nav_eval.enable(s, log, "probe")
        # P: the supported goal nearest the map's centre of free space with room around it.
        pts = nav_eval.sample_supported_goals(s, args.seed + 7, count=6)
        chosen = None
        attempts = []
        for p in pts:
            first = nav_eval.run_goal(s, {"name": "P", "expect": "reach", "map_xy": list(p), "xy": None}, log)
            attempts.append({k: first.get(k) for k in ("result", "navigation_status", "navigation_reason",
                                                       "duration")})
            if not first.get("arrived") or rt.pose is None:
                continue
            th = rt.pose[2] + math.radians(turn)
            d = np.array([math.cos(th), math.sin(th)])
            here = np.asarray(rt.pose[:2], float)
            goal = here + 1.5 * d
            box = here + dist * d
            line = here + np.linspace(0.3, 1.5, 13)[:, None] * d
            if rt.planner.traversable_xy(rt.grid, line).all():
                chosen = (here, goal, box)
                break
        if chosen is None:
            row["result"] = "skipped: no straight certified 1.5 m line"
            row["P_attempts"] = attempts
            row["guard"] = dict(rt.guard.stats) if rt.guard else None
            row["guard_asserted"] = None if rt.guard is None else \
                evaluator.score_guard(s.world, rt.grid, s.origin, rt.guard.asserted_log)
            results.append(row)
            print(row, flush=True)
            s.close()
            continue
        here, goal, box = chosen
        box_world = np.asarray(evaluator.map_to_world(s.origin, box), float)
        s.world.place_evaluation_obstacle(box_world)
        for _ in range(3):
            s.control_step()
        n0 = len(s.contacts)
        trace = []
        if args.trace:
            _step0 = s.control_step

            def _traced(*a, _step0=_step0, _trace=trace, _s=s, _rt=rt, _box=box_world, **k):
                r = _step0(*a, **k)
                tp = evaluator.true_pose(_s.world)
                half = _s.world.evaluation_obstacle_size[0] / 2
                g = _rt.guard
                _trace.append({"t": round(_s.now, 2),
                               "centre_to_box_edge": round(math.hypot(max(abs(tp[0] - _box[0]) - half, 0),
                                                                      max(abs(tp[1] - _box[1]) - half, 0)), 3),
                               "cmd": [round(v, 3) for v in _rt.last_command], "nav": _rt.nav.status,
                               "holding": _rt.nav.holding, "slam": _rt.slam.status,
                               "free_run": None if g is None or g.free_run is None else round(g.free_run, 2)})
                return r

            s.control_step = _traced
        rec = nav_eval.run_goal(s, {"name": "G", "expect": "reach", "map_xy": list(goal), "xy": None}, log,
                                timeout=40.0)
        hit = [c for c in s.contacts[n0:] if "evaluation_obstacle" in c["with"]]
        tp = evaluator.true_pose(s.world)
        half = s.world.evaluation_obstacle_size[0] / 2
        gap = math.hypot(max(abs(tp[0] - box_world[0]) - half, 0), max(abs(tp[1] - box_world[1]) - half, 0))
        row["P_attempts"] = attempts
        row.update({"result": rec.get("result"), "nav": rec.get("navigation_status"),
                    "reason": rec.get("navigation_reason"), "box_contacts": len(hit),
                    "first_contact_t": hit[0]["t"] - rec["t_start"] if hit else None,
                    "revocations": [r["reason"] for r in rec.get("revocations", [])],
                    "final_centre_to_box_edge_m": round(gap, 3),
                    "guard": dict(rt.guard.stats) if rt.guard else None, "guard_status": rt.guard_status,
                    "guard_asserted": None if rt.guard is None else
                    evaluator.score_guard(s.world, rt.grid, s.origin, rt.guard.asserted_log),
                    "asserted_log": None if rt.guard is None else rt.guard.asserted_log})
        results.append(row)
        print(row, flush=True)
        if args.trace:
            (work / "trace.json").write_text(json.dumps(trace))
        s.close()
Path(args.out).write_text(json.dumps(results, indent=1))
n = [r for r in results if "box_contacts" in r]
print(f"SUMMARY ran {len(n)}/{len(results)}; with contact {sum(r['box_contacts'] > 0 for r in n)}")
