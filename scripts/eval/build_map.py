"""Build and save a map by autonomous exploration (onboard RGB only; no interactions).

    PYTHONPATH=src python scripts/eval/build_map.py --world arena --seconds 420
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    dump,
    fresh_dir,
    keyframe_map_error,
    operator_turn_until_tracking,
    provenance,
    score_loops,
    trajectory_metrics,
)

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402
from amr_rl.sim.world import WORLD_DIR, load_world_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="arena")
    parser.add_argument("--seconds", type=float, default=420)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-loop-closure", action="store_true", help="A/B: run without loop closure")
    parser.add_argument("--coverage-seconds", type=float, default=0.0,
                        help="finish with the coverage pass for this many seconds (survey turns far from keyframes)")
    parser.add_argument("--vslam", nargs="*", default=[], help="A/B: VSLAMConfig overrides key=value (JSON values)")
    parser.add_argument("--no-coverage", action="store_true",
                        help="A/B: no coverage pass at all (also not when frontiers run out)")
    parser.add_argument("--loop-shadow", action="store_true",
                        help="evaluation: detect and verify loops but only record them (paired scoring)")
    parser.add_argument("--legacy-loop-candidates", action="store_true",
                        help="A/B: round-5 candidates (top 3 by similarity, no uncertainty slots, no revisits)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else fresh_dir(f"map-{args.world}")
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, configs=[WORLD_DIR / f"{args.world}.yaml"], extra={"args": vars(args)})
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only", seed=args.seed,
                        place_descriptor=None if args.no_loop_closure else "megaloc")
    if args.loop_shadow:  # log more candidates; revisits change the trajectory, so off
        cfg.vslam.loop_shadow, cfg.vslam.loop_top_k, cfg.vslam.loop_uncertain_k, cfg.revisit_sigma = True, 3, 3, 0.0
        cfg.vslam.covis_max_per_kf = 3  # recorded for the offline comparison only
    for item in args.vslam:
        key, value = item.split("=", 1)
        if not hasattr(cfg.vslam, key):
            raise SystemExit(f"unknown VSLAMConfig field {key}")
        setattr(cfg.vslam, key, json.loads(value))
    if args.no_coverage:
        cfg.coverage_lattice = 0.0
    elif args.coverage_seconds > 0:
        cfg.coverage_lattice = 0.35
    if args.legacy_loop_candidates:
        cfg.vslam.loop_top_k, cfg.vslam.loop_uncertain_k, cfg.revisit_sigma = 3, 0, 0.0
    # Seed k turns the configured start heading by k x 72 deg (seed 0 = configured start).
    start = list(load_world_config(args.world)["robot_start"])
    start[2] = float(start[2] + args.seed * 2 * math.pi / 5)
    s = Session(args.world, run_dir=out / "session", memory_path=out / "throwaway-memory.sqlite", config=cfg,
                seed=args.seed, inspection=False, world_overrides={"robot_start": start})
    s.control_step()
    s.enable_autonomy()
    reenable = 0
    turns = []  # operator relocalisation turns (interventions, counted)
    last_tracking = s.now
    while s.now < args.seconds:
        s.control_step()
        rt = s.runtime
        rt.coverage_mode = args.coverage_seconds > 0 and s.now >= args.seconds - args.coverage_seconds
        if rt.slam.status == "tracking":
            last_tracking = s.now
        if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
            reenable += 1
            if reenable > 5:
                break
            s.enable_autonomy()
        elif (not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None
              and s.now - last_tracking >= 5.0 and len(turns) < 5):
            # Bounded recovery gave up: act as the operator (as navigation/learning do).
            operator_turn_until_tracking(s, turns, timeout=min(40.0, max(0.0, args.seconds - s.now)))
        if int(s.now * 10) % 300 == 0:
            print(f"t={s.now:.0f} map={rt.grid.counts()} loc={rt.slam.status}", flush=True)
    meta = s.runtime.save_map(out / "map")
    img, _ = s.runtime.map_image()
    cv2.imwrite(str(out / "map.png"), img[..., ::-1])
    result = {"map_version": meta["map_version"], "landmarks": meta["landmarks"], "reenable_interventions": reenable,
              "operator_turns": turns,
              "trajectory": trajectory_metrics(s.truth),
              "keyframe_map_error": keyframe_map_error(s.runtime.slam.keyframes, s.truth),
              "loop_closure": {"status": s.runtime.place_status, "closures": len(s.runtime.loop_closures),
                               "candidates_checked": len(s.runtime.slam.loop_log),
                               "scored": score_loops(s.runtime.slam.loop_log, s.truth),
                               "log": s.runtime.slam.loop_log,
                               "covis_edges": [[int(a), int(b), [float(v) for v in z], int(n)]
                                               for a, b, z, n in s.runtime.slam.covis_edges]},
              "map_score": evaluator.score_map(s.world, s.runtime.grid, s.origin), "contacts": s.contacts,
              "sim_seconds": s.now,
              "origin_world": [float(v) for v in s.origin]}  # evaluation-only: world pose of the map frame
    dump(out / "result.json", result)
    s.save_summary()
    s.close()
    print(result)
    print(f"MAP_DIR={out / 'map'}")


if __name__ == "__main__":
    main()
