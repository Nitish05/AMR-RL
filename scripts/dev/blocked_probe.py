"""Blocked-robot probe (evaluation only): drive straight into a heavy box the robot does
not avoid (near-field guard off) and record whether, and how fast, the robot notices
that it is not moving: the VSLAM's blocked verdict (since 2026-10-12 reported with the pose
kept, ``info["blocked"]``; before, the motion-consistency loss
``visual_motion_inconsistent_with_commands``) that starts the contact back-off.

One case per run: boot, track for a few seconds, place the box ``--gaps`` metres in front of
the front edge (true pose, evaluation only), then command forward at ``--speed`` for
``--seconds`` with operator drive commands. A ``none`` gap drives on open floor (false
alarm check). Per frame it logs the command, measured odometry (slip flags), the VSLAM
status and reason, the true speed and contacts.

    PYTHONPATH=src python scripts/dev/blocked_probe.py out_dir --world home_a --gap 0.25 --odometry imu_encoders
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import add_runtime_args, apply_runtime_args, provenance, world_overrides  # noqa: E402

from amr_rl.control.supervisor import SupervisorConfig  # noqa: E402
from amr_rl.runtime.robot import RuntimeConfig  # noqa: E402
from amr_rl.sim import evaluator  # noqa: E402
from amr_rl.sim.harness import Session  # noqa: E402


def _xy(entity):
    pos = entity.get_pos()
    pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
    return np.asarray(pos, float).ravel()[:2]


def run_case(args, out, gap, seed):
    cfg = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=False), policy="explore_only", seed=seed,
                        place_descriptor=None)
    apply_runtime_args(cfg, args)
    cfg.near_field_guard = False  # the box must not be avoided: this probes the blind case
    name = out.name
    s = Session(args.world, run_dir=out / "session", memory_path=out / "mem.sqlite", config=cfg,
                seed=seed, inspection=False, world_overrides=world_overrides(args))
    rt = s.runtime
    gen = rt.supervisor.generation

    def drive(v):
        rt.command({"action": "manual", "v": v, "w": 0.0, "generation": rt.supervisor.generation})
        s.control_step()

    while s.now < args.settle:  # boot stand-still, first keyframes
        drive(0.0)
    truth = evaluator.true_pose(s.world)
    half = s.world.spec.chassis["length"] / 2 + s.world.spec.chassis["axle_offset_x"]
    box = None
    if gap != "none":
        size = s.world.evaluation_obstacle_size
        d = half + float(gap) + size[0] / 2
        box = truth[:2] + d * np.array([math.cos(truth[2]), math.sin(truth[2])])
        s.world.place_evaluation_obstacle(box)
        for _ in range(3):
            drive(0.0)
    t0, rows, first_contact, detected = s.now, [], None, None
    prev = evaluator.true_pose(s.world)
    while s.now < t0 + args.seconds:
        drive(args.speed)
        tr = s.truth[-1]
        now_pose = evaluator.true_pose(s.world)
        true_v = float(np.hypot(*(now_pose[:2] - prev[:2]))) / 0.1
        prev = now_pose
        touching = evaluator.robot_contacts(s.world)
        if touching and first_contact is None:
            first_contact = s.now
        row = {"t": round(s.now - t0, 2), "status": tr["status"], "reason": tr.get("reason"), "cmd": tr.get("cmd"),
               "true_v": round(true_v, 3), "odo": tr.get("odo"), "slip": tr.get("odo_slip"),
               "slip_reasons": tr.get("odo_slip_reasons"), "contact": touching,
               "inconsistency": rt.slam.inconsistency}
        rows.append(row)
        if detected is None and ((tr["status"] == "lost" and tr.get("reason") == "visual_motion_inconsistent_with_commands")
                                 or (tr.get("info") or {}).get("blocked") or tr.get("blocked")):
            detected = s.now
    pushed = None if box is None else float(np.hypot(*(_xy(s.world.evaluation_obstacle) - box)))
    res = {"case": name, "world": args.world, "odometry": args.odometry, "gap_m": gap, "seed": seed,
           "first_contact_s": None if first_contact is None else round(first_contact - t0, 2),
           "detected_s": None if detected is None else round(detected - t0, 2),
           "detection_after_contact_s": None if detected is None or first_contact is None
           else round(detected - first_contact, 2),
           "contact_frames": sum(1 for r in rows if r["contact"]), "box_pushed_m": pushed,
           "odometry_status": rt.odometry_status() if rt.odo is not None else None, "generation_start": gen}
    (out / "frames.json").write_text(json.dumps(rows, indent=0, default=float))
    s.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--world", default="home_a")
    ap.add_argument("--gap", default="0.25", help="metres from the front edge to the box, or 'none'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=4.0)
    add_runtime_args(ap)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, extra={"args": vars(args)})
    r = run_case(args, out, args.gap, args.seed)  # one Genesis scene per process
    print(json.dumps({k: v for k, v in r.items() if k != "odometry_status"}), flush=True)
    (out / "result.json").write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
