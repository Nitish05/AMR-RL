"""Controlled A/B of mapper settings on ONE trajectory (evaluation only).

The robot runs with the primary mapper (current defaults). A shadow mapper with the
alternative settings receives the same keyframes into its own grid; every other map
input (footprint, attestation, landmark and fixture evidence, keep-out discs) is
mirrored into both grids. Both maps are scored against ground truth at the end.

    PYTHONPATH=src python scripts/dev/mapper_ab.py heldout_c 240 <run_dir> suppress_free_above_bases=false
"""

import json
import sys

from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.mapping.occupancy import EvidenceConfig, FloorEvidenceMapper, OccupancyGrid
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim import evaluator
from amr_rl.sim.harness import Session

world, seconds, run_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
overrides = {}
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    overrides[k] = {"true": True, "false": False}.get(v.lower(), v)
    if isinstance(overrides[k], str):
        overrides[k] = float(v) if "." in v else int(v)

cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only")
s = Session(world, run_dir=run_dir, memory_path=run_dir + "/m.sqlite", config=cfg, inspection=False)
rt = s.runtime
g1 = rt.grid
g2 = OccupancyGrid(g1.map_version, g1.cfg)
m2 = FloorEvidenceMapper(rt.model, g2, EvidenceConfig(**{**EvidenceConfig().__dict__, **overrides}))
state = {"primary": False}
orig_add, orig_keep = g1.add_hits, g1.set_keepout


def add_hits(xy, delta, *, min_count=1):
    if not state["primary"]:
        g2.add_hits(xy, delta, min_count=min_count)
    return orig_add(xy, delta, min_count=min_count)


def set_keepout(discs):
    g2.set_keepout(discs)
    return orig_keep(discs)


g1.add_hits, g1.set_keepout = add_hits, set_keepout
orig_ukf = rt.mapper.update_keyframe


def update_keyframe(keyframes, *a, **k):
    state["primary"] = True
    try:
        out = orig_ukf(keyframes, *a, **k)
    finally:
        state["primary"] = False
    m2.update_keyframe(keyframes, *a, **k)
    return out


rt.mapper.update_keyframe = update_keyframe
s.control_step()
s.enable_autonomy()
while s.now < seconds:
    s.control_step()
    if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
        s.enable_autonomy()
res = {"primary (defaults)": evaluator.score_map(s.world, g1, s.origin),
       f"shadow {overrides}": evaluator.score_map(s.world, g2, s.origin)}
print(json.dumps(res, indent=1))
