"""Localisation benchmark for VSLAM variants: exploration runs scored against ground
truth (evaluation only). Usage:

    PYTHONPATH=src python scripts/dev/slam_bench.py <variant> <scenario>... --seconds 300 --out <dir>

variant: old | new | cmdscore ; scenario: <world>:<seed> (seed k turns the start by k x 72 deg)
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import trajectory_metrics  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.perception.vslam import VSLAMConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402
from amr_rl.sim.world import load_world_config  # noqa: E402

VARIANTS = {
    "old": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=False),
    "new": VSLAMConfig(static_hypothesis_when_moving=False, command_consistency_score=True),
    "cmdscore": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=True),
    "reloc": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=False,
                         reloc_min_inliers=60, reloc_confirmations=3),
    "pre": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=False,
                       freeze_check=False, purge_on_loss=False, dr_gate=False),
    "local": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=False,
                         reloc_local_first=True),
    "cmdreloc": VSLAMConfig(static_hypothesis_when_moving=True, command_consistency_score=True,
                            reloc_min_inliers=60, reloc_confirmations=3),
}

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("scenarios", nargs="+")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for scen in a.scenarios:
    world, seed = scen.split(":")
    seed = int(seed)
    start = list(load_world_config(world)["robot_start"])
    start[2] += seed * 2 * math.pi / 5
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only",
                        vslam=VARIANTS[a.variant], seed=seed)
    run = out / f"{a.variant}-{world}-s{seed}"
    s = Session(world, run_dir=run, memory_path=run / "m.sqlite", config=cfg, inspection=False,
                world_overrides={"robot_start": start})
    s.control_step()
    s.enable_autonomy()
    reenable = 0
    while s.now < a.seconds:
        s.control_step()
        rt = s.runtime
        if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
            reenable += 1
            s.enable_autonomy()
    t = trajectory_metrics(s.truth)
    res = {"variant": a.variant, "world": world, "seed": seed, "ate": t["ate_rmse_m"], "max": t["max_position_error_m"],
           "heading_max": t["max_heading_error_deg"], "lost": t["lost_frames"], "path": t["true_path_length_m"],
           "reenable": reenable, "coverage": evaluator.score_map(s.world, s.runtime.grid, s.origin)["free_coverage"],
           "frozen_events": len(s.runtime.slam.frozen_events), "contacts": len(s.contacts)}
    print(json.dumps(res), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    s.close()
