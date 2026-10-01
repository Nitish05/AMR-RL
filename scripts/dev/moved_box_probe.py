"""Timeline of the obstacle-on-route fault (evaluation probe).

Loads the map a navigation run saved, starts the robot where that run started,
relocalises, then runs the same fault test as scripts/eval/navigation.py (box placed
on the robot's own planned route). Every control step logs the true gap between the
robot's front and the box, the navigator state, the corridor check, and the map
evidence over the box footprint. Ground truth is used for logging only.

    PYTHONPATH=src python scripts/dev/moved_box_probe.py <nav run dir> <out dir> [seed]
"""

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

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
out.mkdir(parents=True, exist_ok=True)
res = json.loads((run_dir / "result.json").read_text())
start = res["start_world"]
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only", seed=seed)
s = Session(res["world"], run_dir=str(out), memory_path=str(out / "m.sqlite"), config=cfg, seed=seed,
            inspection=False, map_dir=str(run_dir / "map"), map_origin=start,
            world_overrides={"robot_start": start})
rt = s.runtime
log = []
if not nav_eval.wait_tracking(s, 40, log):
    print("never relocalised", flush=True)
    sys.exit(1)
rt.chooser.policy = "operator_only"
nav_eval.enable(s, log, "probe")

rows = []
orig_step = s.control_step


def box_cells_logodds():
    if not s.world.static_geometry or s.world.static_geometry[-1]["name"] != "evaluation_obstacle":
        return None
    g = s.world.static_geometry[-1]
    half = g["size"][0] / 2
    xs = np.linspace(-half, half, 8)
    pts = np.array([[g["xy"][0] + a, g["xy"][1] + b] for a in xs for b in xs])
    m = np.array([evaluator.world_to_map(s.origin, p) for p in pts])
    ix, iy = rt.grid.to_cell(m)
    ok = rt.grid.inside(ix, iy)
    lo = rt.grid.logodds[iy[ok], ix[ok]]
    return float(lo.max()), float(np.mean(lo > 0.3))


def logged_step(*a, **k):
    r = orig_step(*a, **k)
    geo = s.world.static_geometry[-1] if s.world.static_geometry else None
    if geo and geo["name"] == "evaluation_obstacle":
        tp = evaluator.true_pose(s.world)
        half = geo["size"][0] / 2
        dx = max(abs(tp[0] - geo["xy"][0]) - half, 0.0)
        dy = max(abs(tp[1] - geo["xy"][1]) - half, 0.0)
        gap = math.hypot(dx, dy) - 0.18  # robot half-diagonal (~0.23) minus margin, rough
        lo = box_cells_logodds()
        n = rt.nav
        corridor = n._corridor_clear(rt.grid, rt.pose) if (rt.pose is not None and len(n.path)) else None
        rows.append({"t": round(s.now, 2), "centre_gap": round(math.hypot(dx, dy), 3), "approx_gap": round(gap, 3),
                     "nav": n.status, "holding": n.holding, "corridor_ok": corridor,
                     "box_logodds_max": None if lo is None else round(lo[0], 2),
                     "box_frac_occ": None if lo is None else round(lo[1], 2),
                     "slam": rt.slam.status, "cmd": [round(v, 3) for v in rt.last_command],
                     "contact": any("evaluation_obstacle" in c["with"] and abs(c["t"] - s.now) < 0.05
                                    for c in s.contacts[-3:])})
    return r


s.control_step = logged_step
record = nav_eval.fault_test(s, log, seed + 202, "obstacle_placed_on_mapped_route", obstacle=True)
(out / "timeline.json").write_text(json.dumps({"record": {k: v for k, v in record.items() if k != "contacts"},
                                               "rows": rows}, indent=1, default=str))
print("result", record.get("result"), record.get("reason"), "contact", record.get("obstacle_contact"))
last = None
for r in rows:
    key = (r["nav"], r["holding"], r["corridor_ok"], r["slam"], r["contact"])
    if key != last or int(r["t"] * 10) % 10 == 0:
        print(r)
        last = key
