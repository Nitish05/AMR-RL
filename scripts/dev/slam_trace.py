"""Replay an exploration run and print VSLAM status/reason changes in a time window,
with the evaluator's true error (scoring only) and relocalisation gate state.

    PYTHONPATH=src python scripts/dev/slam_trace.py arena 250 272 <run_dir>
"""

import math
import sys

import numpy as np

from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim import evaluator
from amr_rl.sim.harness import Session

world, t0, t1, run_dir = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only")
s = Session(world, run_dir=run_dir, memory_path=run_dir + "/m.sqlite", config=cfg, inspection=False)
rt = s.runtime
s.control_step()
s.enable_autonomy()
last = None
while s.now < t1:
    s.control_step()
    if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
        s.enable_autonomy()
    if s.now < t0:
        continue
    r = rt.last_track
    truth = evaluator.to_map_frame(s.origin, evaluator.true_pose(s.world))
    err = None if r is None or r.pose is None else float(np.hypot(*(r.pose[:2] - truth[:2])))
    herr = None if r is None or r.pose is None else math.degrees(math.atan2(math.sin(r.pose[2] - truth[2]),
                                                                            math.cos(r.pose[2] - truth[2])))
    key = (r.status if r else None, r.reason if r else None, rt.slam.last_hypothesis)
    if key != last or int(s.now * 10) % 5 == 0:
        dr = rt.slam._dr
        print(f"{s.now:6.1f} {key} inl={r.inliers if r else None} err={err and round(err, 3)} "
              f"herr={herr and round(herr, 1)} truth={np.round(truth, 2).tolist()} "
              f"dr={None if dr is None else np.round(dr['pose'], 2).tolist()} "
              f"act={rt.activity.name if rt.activity else None} cmd={rt.last_command}", flush=True)
        last = key
print("frozen_events", rt.slam.frozen_events)
