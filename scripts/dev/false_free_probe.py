"""Evaluation-only diagnostic: which mapping step puts FREE evidence into cells that are
truly inside obstacles (false free)? Tags free hits by source: parallax floor evidence
('floor'), the robot's own footprint ('footprint'), start attestation ('attest').

    PYTHONPATH=src python scripts/dev/false_free_probe.py heldout_c 200 <run_dir>
"""

import json
import sys
from collections import defaultdict

import numpy as np

from amr_rl.control.supervisor import SupervisorConfig
from amr_rl.runtime.robot import RuntimeConfig
from amr_rl.sim import evaluator
from amr_rl.sim.harness import Session

world, seconds, run_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only")
s = Session(world, run_dir=run_dir, memory_path=run_dir + "/m.sqlite", config=cfg, inspection=False)
rt, grid = s.runtime, s.runtime.grid
truth = evaluator.true_obstacle_mask(s.world, grid, s.origin, include_fixtures=True)
counts = defaultdict(lambda: [0, 0])  # source -> [free hits, hits inside true obstacles]
tag = {"src": None}
orig_add = grid.add_hits


def add_hits(xy, delta, *, min_count=1):
    if delta < 0 and len(xy):
        ix, iy = grid.to_cell(np.asarray(xy, float))
        ok = grid.inside(ix, iy)
        bad = truth[iy[ok], ix[ok]]
        c = counts[tag["src"] or "other"]
        c[0] += int(ok.sum())
        c[1] += int(bad.sum())
    return orig_add(xy, delta, min_count=min_count)


def wrap(name, fn):
    def inner(*a, **k):
        tag["src"] = name
        try:
            return fn(*a, **k)
        finally:
            tag["src"] = None
    return inner


grid.add_hits = add_hits
rt.mapper.update = wrap("floor", rt.mapper.update)
grid.mark_footprint = wrap("footprint", grid.mark_footprint)
grid.mark_free_disc = wrap("attest", grid.mark_free_disc)
s.control_step()
s.enable_autonomy()
while s.now < seconds:
    s.control_step()
    if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
        s.enable_autonomy()
cls = grid.classes()
print(json.dumps({"free_hits_by_source": {k: {"hits": v[0], "inside_true_obstacles": v[1]} for k, v in counts.items()},
                  "false_free_cells": int(((cls == 1) & truth).sum()),
                  "score": evaluator.score_map(s.world, grid, s.origin), "contacts": len(s.contacts)}, indent=1))
