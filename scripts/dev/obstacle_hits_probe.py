"""Evaluation-only diagnostic: where does obstacle evidence come from, and how much
of it lands on true free floor? Wraps the mapper to tag each obstacle hit by
source (parallax 'bases' or triangulated 'landmarks') and scores the hit cells
against the true obstacle mask (ground truth used for scoring only).

    PYTHONPATH=src python scripts/dev/obstacle_hits_probe.py arena 200 <run_dir>
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
rt = s.runtime
grid = rt.grid
log = defaultdict(list)  # source -> list of (ix, iy, t, activity)
tag = {"src": None}
orig_add = grid.add_hits


def add_hits(xy, delta, *, min_count=1):
    if delta > 0 and len(xy):
        ix, iy = grid.to_cell(np.asarray(xy, float))
        act = rt.activity.name if rt.activity else None
        phase = rt.activity.phase if rt.activity else None
        for a, b in zip(ix, iy):
            log[tag["src"] or "other"].append((int(a), int(b), s.now, f"{act}:{phase}"))
    return orig_add(xy, delta, min_count=min_count)


grid.add_hits = add_hits
orig_update = rt.mapper.update
orig_lm = rt.mapper.add_landmark_obstacles


def update(ref, cur, **kw):
    tag["src"] = "bases"
    try:
        return orig_update(ref, cur, **kw)
    finally:
        tag["src"] = None


def add_lm(points, low=0.04, high=0.45):
    tag["src"] = "landmarks"
    try:
        return orig_lm(points, low=low, high=high)
    finally:
        tag["src"] = None


orig_commit = rt.mapper._commit_confirmed_bases


def commit(results):
    tag["src"] = "bases"
    try:
        return orig_commit(results)
    finally:
        tag["src"] = None


rt.mapper.update = update
rt.mapper.add_landmark_obstacles = add_lm
rt.mapper._commit_confirmed_bases = commit
s.control_step()
s.enable_autonomy()
while s.now < seconds:
    s.control_step()
    if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
        s.enable_autonomy()
truth = evaluator.true_obstacle_mask(s.world, grid, s.origin)
near = evaluator.true_obstacle_mask(s.world, grid, s.origin, margin=0.10)
report = {}
for src, hits in log.items():
    arr = np.array([(a, b) for a, b, _, _ in hits])
    on_obstacle = truth[arr[:, 1], arr[:, 0]]
    near_obstacle = near[arr[:, 1], arr[:, 0]]
    stray = ~near_obstacle
    by_act = defaultdict(lambda: [0, 0])
    for (_a, _b, _t, act), st in zip(hits, stray):
        by_act[act][0] += 1
        by_act[act][1] += int(st)
    cells = {(a, b) for a, b, _, _ in hits}
    stray_cells = {(a, b) for (a, b, _, _), st in zip(hits, stray) if st}
    report[src] = {"hits": len(hits), "on_obstacle": int(on_obstacle.sum()),
                   "stray_hits_gt10cm_from_obstacle": int(stray.sum()),
                   "cells": len(cells), "stray_cells": len(stray_cells),
                   "by_activity": {k: {"hits": v[0], "stray": v[1]} for k, v in sorted(by_act.items())}}
cls = grid.classes()
report["mapper_stats"] = {k: v for k, v in rt.mapper.stats.items() if k.startswith("bases")}
report["final"] = {"counts": grid.counts(), "score": evaluator.score_map(s.world, grid, s.origin),
                   "occupied_cells_on_true_free_far": int(((cls == 2) & ~near).sum())}
print(json.dumps(report, indent=1))
s.close()
