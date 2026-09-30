"""Dev probe: force one engage on the nearest visible entity and trace world/robot."""
import math
import sys

import cv2
import numpy as np

from amr_rl.behavior.activities import Engage
from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim.harness import Session

target_name, action = sys.argv[1], sys.argv[2]
out = f"/tmp/claude-0/scr/probe_{target_name}_{action}"
import shutil; shutil.rmtree(out, ignore_errors=True)
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), initial_survey=False)
cfg.chooser.explore_weight = 0.0
s = Session("dev_interact", run_dir=out, memory_path=f"{out}/m.sqlite", config=cfg)
item, ent = s.world.fixtures[target_name]
fx, fy = item["pos"]
# place robot 1.3 m from target facing it (reset only)
ang = math.atan2(-0.2 - fy, -1.0 - fx)
D = 0.75 if action == 'signal' else 0.52
s.world.backend.initialize_pose(fx + D * math.cos(ang), fy + D * math.sin(ang), ang + math.pi)
for _ in range(30): s.world.step()
s.origin = __import__("amr_rl.sim.evaluator", fromlist=["x"]).true_pose(s.world)
for _ in range(15): s.control_step()
rt = s.runtime
print("entities", [(e["entity_id"], e["appearance"]["color"], round(e["x"],2), round(e["y"],2)) for e in rt.known_entities()])
s.enable_autonomy(); s.control_step()
color = {"bloom": "cyan", "grump": "green", "roller": "magenta"}[target_name]
eid = [e for e in rt.known_entities() if e["appearance"]["color"] == color][0]
act = Engage(eid["entity_id"], action, (eid["x"], eid["y"]), {"activity": "engage", "value": 0, "basis": "probe"})
act.since = s.now; rt.activity = act; rt._survey_done = True
rt.chooser.decide = lambda *a, **k: None
for i in range(400):
    s.control_step()
    a = rt.activity
    if i % 5 == 0 or a is None:
        vis = [(v.detection.color, v.detection.state_token, v.entity_id is not None) for v in rt.tracker.visible]
        print(round(s.now,1), None if a is None else a.phase, rt.last_command, "signal", s.world.signal_active, vis, flush=True)
    if a is not None and a.phase == "acting" and not hasattr(a, "_dbg"):
        a._dbg = 1; print("creep", a.creep, "range", a.before[-1]["range"], "app", rt.entity_appearance(a.target_entity))
    if i % 5 == 0:
        from amr_rl.sim import evaluator as ev
        print("   true robot", np.round(ev.true_pose(s.world),3), "target", np.round(ev.fixture_true_xy(s.world, target_name),3), "contacts", ev.robot_contacts(s.world))
    tr = rt.last_track
    if a is not None and a.phase in ("acting",) or (tr is not None and tr.status != "tracking"):
        print("  TRACK", round(s.now,1), tr.status, tr.inliers, tr.matched, tr.reason, rt.slam.last_hypothesis, len(rt.slam.lm), flush=True)
    if tr is not None and s.now > 7.0 and s.now < 7.75: cv2.imwrite(f"{out}/t{s.now:.1f}.png", rt.last_frame.rgb[..., ::-1])
    if a is None: break
    if i in (60, 90): cv2.imwrite(f"{out}/frame{i}.png", rt.last_frame.rgb[..., ::-1])
print("log", rt.interaction_log)
print("events", [e.__dict__ for e in s.world.events])
print("outcomes", rt.recent_outcomes)
cv2.imwrite(f"{out}/last.png", rt.last_frame.rgb[..., ::-1])
