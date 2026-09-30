"""Build and save a map by autonomous exploration (onboard RGB only; no interactions).

    PYTHONPATH=src python scripts/eval/build_map.py --world arena --seconds 420
"""

import argparse
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump, fresh_dir, provenance, trajectory_metrics  # noqa: E402

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
    args = parser.parse_args()
    out = Path(args.out) if args.out else fresh_dir(f"map-{args.world}")
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, configs=[WORLD_DIR / f"{args.world}.yaml"], extra={"args": vars(args)})
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only", seed=args.seed)
    # Seed k turns the configured start heading by k x 72 deg (seed 0 = configured start).
    start = list(load_world_config(args.world)["robot_start"])
    start[2] = float(start[2] + args.seed * 2 * math.pi / 5)
    s = Session(args.world, run_dir=out / "session", memory_path=out / "throwaway-memory.sqlite", config=cfg,
                seed=args.seed, inspection=False, world_overrides={"robot_start": start})
    s.control_step()
    s.enable_autonomy()
    reenable = 0
    while s.now < args.seconds:
        s.control_step()
        rt = s.runtime
        if not rt.supervisor.autonomy_enabled and rt.supervisor.recovery is None and rt.slam.status == "tracking":
            reenable += 1
            if reenable > 5:
                break
            s.enable_autonomy()
        if int(s.now * 10) % 300 == 0:
            print(f"t={s.now:.0f} map={rt.grid.counts()} loc={rt.slam.status}", flush=True)
    meta = s.runtime.save_map(out / "map")
    img, _ = s.runtime.map_image()
    cv2.imwrite(str(out / "map.png"), img[..., ::-1])
    result = {"map_version": meta["map_version"], "landmarks": meta["landmarks"], "reenable_interventions": reenable,
              "trajectory": trajectory_metrics(s.truth),
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
