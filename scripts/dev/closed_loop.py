"""Development closed-loop run (not an evaluation)."""
import argparse
import json
import time

import cv2

from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim.harness import Session

p = argparse.ArgumentParser()
p.add_argument("--world", default="home_a"); p.add_argument("--seconds", type=float, default=120)
p.add_argument("--out", default="/tmp/claude-0/scr/cl"); p.add_argument("--policy", default="learned")
p.add_argument("--memory", default=None); p.add_argument("--map", default=None)
p.add_argument("--save-map", default=None); p.add_argument("--explore-weight", type=float, default=None)
a = p.parse_args()
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy=a.policy)
if a.explore_weight is not None: cfg.chooser.explore_weight = a.explore_weight
s = Session(a.world, run_dir=a.out, memory_path=a.memory or f"{a.out}/memory.sqlite", config=cfg, map_dir=a.map)
s.control_step(); s.control_step()
s.enable_autonomy()
last = [0.0]
def cb(sess):
    st = sess.runtime.last_state
    if sess.now - last[0] >= 5.0:
        last[0] = sess.now
        t = sess.truth[-1]
        print(f"t={sess.now:6.1f} loc={st['localization']['status']:9s} err={t['err'] if t['err'] is None else round(t['err'],3)} "
              f"act={st['activity']['name']}:{st['activity']['phase']} auth={st['authority']['autonomy_enabled']}/{st['authority']['revoked_reason']} "
              f"map={st['map']['free']}/{st['map']['occupied']} ents={len([e for e in st['entities'] if e['entity_id']])} "
              f"need={st['motivation']['stimulation_need']:.2f} out={st['memory']['outcomes']} wall={time.time()-sess.wall_start:.0f}", flush=True)
s.run(a.seconds, callback=cb)
summ = s.save_summary()
img, meta = s.runtime.map_image(); cv2.imwrite(f"{a.out}/map.png", img[..., ::-1])
ins = s.world.render_inspection(); cv2.imwrite(f"{a.out}/inspect.png", ins[..., ::-1])
cv2.imwrite(f"{a.out}/onboard.png", s.runtime.last_frame.rgb[..., ::-1])
if a.save_map: s.runtime.save_map(a.save_map)
print(json.dumps({k: summ[k] for k in ("sim_seconds","wall_seconds","tracking_fraction","ate_rmse","max_position_error","map_score","memory")}, indent=1))
print("contacts", summ["contact_episodes"][:10])
print("activities", [ (round(x['t'],1), x['activity'], x.get('status'), x.get('reason'), x.get('observed')) for x in summ["activities"]][-30:])
print("events", summ["world_events"])
st = s.runtime.last_state
print("DECISION", json.dumps(st["decision"], indent=0, default=str)[:3000])
print("ENTITIES", json.dumps(st["entities"], indent=0, default=str)[:3000])
s.close()
