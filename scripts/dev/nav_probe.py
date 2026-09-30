"""Diagnostic: run the learned policy on a saved map and print every navigator
status/reason change (the evidence recorder keeps only status and goal)."""

import sys

from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim.harness import Session

map_dir, seconds = sys.argv[1], float(sys.argv[2])
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="learned", initial_survey=False)
s = Session("arena", run_dir=sys.argv[3], memory_path=sys.argv[3] + "/m.sqlite", config=cfg, inspection=False,
            map_dir=map_dir, consequences={})
rt = s.runtime
while s.now < 30 and rt.slam.status != "tracking":
    s.control_step()
rt.command({"action": "enable_autonomy", "generation": rt.supervisor.generation})
last = None
while s.now < seconds:
    s.control_step()
    key = (rt.activity.name if rt.activity else None, rt.nav.status, rt.nav.reason, getattr(rt.nav, "holding", None))
    if key != last:
        pose = None if rt.pose is None else [round(float(v), 2) for v in rt.pose]
        print(f"{s.now:6.1f} {key} replans={rt.nav.replans} pose={pose} goal={rt.nav.goal}", flush=True)
        last = key
